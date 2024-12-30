#!/usr/bin/env python3
import os
import gpus
from tqdm import tqdm
from glob import glob
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pickle
import torch
import librosa

# Load the OpenAI "medium" Whisper model and processor
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
#model = WhisperForConditionalGeneration.from_pretrained(model_name)

def extract_folder (folder_path, output_path):
    pool = {}
    for path in tqdm(list(glob(folder_path + "/*.mp3"))):
        key = os.path.basename(path).split(".")[0]
        audio, _ = librosa.load(path, sr=16000)  # Whisper models expect 16kHz audio
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        pool[key] = input_features
    with open(output_path, "wb") as f:
        pickle.dump(pool, f)

extract_folder("data/train_audios", "data/train_features.pkl")
extract_folder("data/test_audios", "data/test_features.pkl")

