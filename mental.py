#!/usr/bin/env python3
import sys
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers.models.whisper.modeling_whisper import WhisperEncoder, BaseModelOutput
from transformers.models.whisper import WhisperPreTrainedModel, WhisperModel
from transformers.models.llama.modeling_llama import SequenceClassifierOutputWithPast

MAX_LENGTH = 3000
LENGTH = 3000
STEP = 200

def whisper_forward (
    self,
    input_features,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    query=None
):
    expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    if input_features.shape[-1] != expected_seq_length:
        raise ValueError(
            f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    embed_pos = self.embed_positions.weight

    hidden_states = inputs_embeds + embed_pos
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    if not query is None:
        # query is [N, C]
        query = torch.unsqueeze(query, 0).expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([hidden_states, query], dim=1)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        if to_drop:
            layer_outputs = (None, None)
        else:
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    None,
                    (head_mask[idx] if head_mask is not None else None),
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)
    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )

class MentalModel (WhisperPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.K = 4
        self.encoder = WhisperEncoder(config)
        self.encoder.forward = whisper_forward.__get__(self.encoder, WhisperEncoder)
        self.embedding = nn.Linear(config.d_model, self.K, dtype=config.torch_dtype, bias=False)
        self.lm_head = nn.Sequential(
                        nn.Linear(config.d_model * self.K + 2, 512, bias=False),
                        nn.ReLU(),
                        nn.Linear(512, 3, bias=False)
                    )
        
        #nn.Linear(384 * self.K + 2, 3, dtype=config.torch_dtype, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.phase = 2      # everything can be trained by default
        self.post_init()

    def init_new_weights (self):
        nn.init.xavier_uniform_(self.lm_head[0].weight)
        nn.init.xavier_uniform_(self.lm_head[2].weight)
        nn.init.xavier_uniform_(self.embedding.weight)

    def set_phase (self, phase):
        self.phase = phase
        if phase == 1:      # train only prediction heads
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        elif phase == 2:    # train all
            for name, param in self.encoder.named_parameters():
                param.requires_grad = True
            pass
        else:
            assert False

    def head_parameters (self):
        p = []
        p.extend(list(self.lm_head.parameters()))
        p.extend(list(self.embedding.parameters()))
        return p

    def parameters (self, *kargs, **kwargs):
        if self.phase == 1:
            return self.head_parameters()
        else:
            return super().parameters(*kargs, **kwargs)

    def forward(self, input_ids, labels = None):        
        feature, meta = input_ids
        X = self.encoder(feature, query=self.embedding.weight)
        X = X.last_hidden_state[:, -self.K:, :]
        #X = torch.amax(X, dim=1)
        X = torch.flatten(X, 1)
        X = torch.cat([X, meta], dim=1)
        logits = self.lm_head(X)
        if labels is None:
            loss = None
        else:
            loss = self.loss_fn(logits, labels)
        return SequenceClassifierOutputWithPast(loss=loss, logits=logits)
