import os
import torch
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import torchvision.models as models
from dataset import create_dataloader
from sklearn.model_selection import train_test_split
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, DeiTImageProcessor

def load_processors():
    ## 이미지 프로세서와 토크나이저 사전 훈련 모델이 사용한 것과 똑같은 것 가져오기.
    DEIT = DeiTImageProcessor.from_pretrained("team-lucid/trocr-small-korean")
    ROBERTA = AutoTokenizer.from_pretrained("team-lucid/trocr-small-korean")
    processor = TrOCRProcessor(image_processor = DEIT, tokenizer = ROBERTA)
    return processor

def create_model(filepath, processor):
    model = VisionEncoderDecoderModel.from_pretrained("team-lucid/trocr-small-korean")
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    save_model(model, filepath)
    print(f'Model created and saved to {filepath}')

def save_model(model, filepath):
    torch.save(model, filepath)
    print(f'Model saved to {filepath}')

def load_model(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(filepath, map_location=device)
    model.to(device)
    print(f'Model loaded from {filepath}')
    return model