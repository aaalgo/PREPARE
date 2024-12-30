#!/usr/bin/env python3
import os, sys
os.environ['WANDB_DISABLED'] = 'true'
from types import SimpleNamespace
import shutil
import types
import gpus
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AdamW, get_scheduler, AutoConfig
from mental import *

meta_lookup = {}

for _, row in pd.read_csv('data/metadata.csv').iterrows():
    uid = row['uid']
    age = row['age']
    gender = 0
    if row['gender'] == 'female':
        gender = 1
    meta_lookup[uid] = [age, gender]

class MentalDataset(Dataset):
    def __init__(self, features_lookup, csv_path, is_train = False):
        self.is_train = is_train
        df = pd.read_csv(csv_path)
        is_submit = not ('label' in df.columns)
        data = []
        for _, row in df.iterrows():
            uid = row['uid']
            if is_submit:
                label = 0
            else:
                label = row['label']
            features = features_lookup[uid]
            data.append((uid, features, label))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, features, label = self.data[idx]
        meta = torch.Tensor([meta_lookup[uid]]).float()
        if self.is_train:
            #off = random.randrange(MAX_LENGTH - LENGTH)
            #features = features[:, :, off:off+LENGTH]
            # augment
            B, D, S = features.shape
            random_augmentation = torch.empty(B, D, S).uniform_(0.95, 1.05)
            features = features * random_augmentation

        return {
            'input_ids': (features, meta),
            'labels': label
        }

def collate_fn(batch):
    # Pad input_ids to the maximum length in the batch        
    featuers = []
    labels = []
    meta = []

    for item in batch:
        ft, mt = item['input_ids']
        featuers.append(ft)
        meta.append(mt)
        labels.append(item['labels'])
    
    featuers_tensor = torch.cat(featuers, dim = 0)
    meta_tensor = torch.cat(meta, dim=0).float()
    labels_tensor = torch.Tensor(labels).long()

    return {
        'input_ids': (featuers_tensor, meta_tensor),
        'labels': labels_tensor
    }

def convert_from_pretrained(args): # Prune is not working
    config = AutoConfig.from_pretrained(args.base_model)
    config.max_source_positions = LENGTH // 2
    model = MentalModel(config)
    pretrained = WhisperModel.from_pretrained(args.base_model)
    state_dict = pretrained.state_dict()
    if config.max_source_positions != 1500:
        state_dict['encoder.embed_positions.weight'] = state_dict['encoder.embed_positions.weight'][:config.max_source_positions, :]
    model.load_state_dict(state_dict, strict=False)
    model.init_new_weights()
    del pretrained
    return model

def train_phase_1 (model, args, train_dataset):
    if args.init_epochs <= 0:
        print("Skipping phase 1 training")
        return
    model.set_phase(1)
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size=args.batch,  # Batch size per device
        num_train_epochs=args.init_epochs,
        warmup_steps=0,
        weight_decay=args.weight_decay,               # Weight decay
        logging_strategy='epoch',
        eval_strategy="no",
        save_strategy="no",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=training_args.max_steps
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()
    pass

def save_predictions(model, path, data):
    #predictions = model.predict(dataset)
    outs = [None]
    for dataset in [data.submit]:
        out = []
        with torch.no_grad():
            for item in dataset:
                features, meta = item['input_ids']
                features = features.to(model.lm_head[0].weight.device)
                meta = meta.to(model.lm_head[0].weight.device)
                logits = model.forward([features, meta]).logits.detach().cpu()
                out.append(logits)
        out = torch.cat(out, dim=0)
        outs.append(out)
    with open(path, 'wb') as f:
        pickle.dump(outs, f)
    print("saved to", path)

def load_data (args):
    data = SimpleNamespace()
    with open('data/train_features.pkl', 'rb') as f:
        train_features = pickle.load(f)
    with open('data/test_features.pkl', 'rb') as f:
        test_features = pickle.load(f)
    data.train = MentalDataset(train_features, f'data/split{args.split}/train_labels.csv', True)
    data.val = MentalDataset(train_features, f'data/split{args.split}/val_labels.csv')
    #data.test = MentalDataset(train_features, 'data/validation_labels.csv')
    data.submit = MentalDataset(test_features, 'data/submission_format.csv')
    del train_features
    del test_features
    return data

def train (args):
    model = convert_from_pretrained(args)
    data = load_data(args)
    print("Phase 1 training")
    train_phase_1(model, args, data.train)
    print("Phase 2 training")
    model.set_phase(2)
    best_save = None
    best_test_loss = 1e6
    best_test_step = 0

    def save_model_wrapper(self, output_dir, _internal_call: bool = False):
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        if self.state.global_step < 60:
            return
        # we always have to create the output_dir or there's nowhere to save the
        # optimizer state (which we don't really need but I don't know how to disable it
        nonlocal best_save
        nonlocal best_test_loss
        nonlocal best_test_step
        # Get the current evaluation loss
        eval_loss = self.state.log_history[-1].get('eval_loss')
        # Compare with the best test loss
        if best_test_loss is None or eval_loss < best_test_loss:
            if best_save is not None and os.path.exists(best_save):
                os.remove(best_save)
            print(f"\nLoss ({eval_loss:.4f}) < {best_test_loss:.4f} at step {best_test_step}.")
            best_save = os.path.join(output_dir, "predict.pkl")
            save_predictions(self.model, best_save, data)
            best_test_loss = eval_loss
            best_test_step = self.state.global_step
        else:
            print(f"\nLoss ({eval_loss:.4f}) did not improve {best_test_loss:.4f} at step {best_test_step}.")

    # Step 6: Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,          # Output directory
        overwrite_output_dir=True,       # Overwrite the output directory if exists
        per_device_train_batch_size=args.batch,  # Batch size per device
        per_device_eval_batch_size=args.batch,   # Evaluation batch size
        #num_train_epochs=args.epochs,
        max_steps=128,
        warmup_steps=0,                # Number of warmup steps
        weight_decay=args.weight_decay,               # Weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_strategy='steps',
        eval_steps=5,
        save_steps=5,
        eval_strategy="steps",
        save_strategy="steps",
        save_only_model=True,
        learning_rate=args.learning_rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data.train,
        eval_dataset=data.val,
        data_collator=collate_fn
    )
    trainer.save_model = types.MethodType(save_model_wrapper, trainer)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, saving model...")
    finally:        
        pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--base_model', type=str, default='openai/whisper-tiny')
    parser.add_argument('--base_model', type=str, default='openai/whisper-medium')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('-s', '--split', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_tmpl', type=str, default='{tag}_s{split}')
    #parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--init_epochs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('-r', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('--tag', type=str, default='model')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.save_tmpl.format(tag=args.tag, split=args.split)

    train(args)

