import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, DeiTImageProcessor

def load_processors():
    ## 이미지 프로세서와 토크나이저 사전 훈련 모델이 사용한 것과 똑같은 것 가져오기.
    DEIT = DeiTImageProcessor.from_pretrained("team-lucid/trocr-small-korean")
    ROBERTA = AutoTokenizer.from_pretrained("team-lucid/trocr-small-korean")
    processor = TrOCRProcessor(image_processor = DEIT, tokenizer = ROBERTA)
    return processor

def create_model(filepath):
    model = VisionEncoderDecoderModel.from_pretrained("team-lucid/trocr-small-korean")
    save_model(model, filepath)

def save_model(model, filepath):
    torch.save(model, filepath)
    print(f'Model saved to {filepath}')

def load_model(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(filepath, map_location=device)
    print(f'Model loaded from {filepath}')
    model.to(device)
    return model

def dataloader(image_dir, label_dir, processor):
    df = pd.read_csv(label_dir, encoding='utf-8')
    ## 데이터 스플릿
    train_df, test_df = train_test_split(df, test_size=0.2, random_state = 0)
    ## 인덱스 초기화
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    ## 데이터셋 클래스에 먹여주기
    train_dataset = HandWriting(image_dir=image_dir, df=train_df, processor=processor)
    test_dataset = HandWriting(image_dir=image_dir, df=test_df, processor=processor)
    ## 데이터 로더
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10)
    return train_dataloader, test_dataloader

class HandWriting(Dataset):
    def __init__(self, image_dir, df, processor, max_target_length=50):
        self.image_dir = image_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        ## 한줄에 많은 수의 단어가 들어오진 않으니
        ## 적당하게 50토큰 정도로 지정해놓자. (대략 30~40 단어)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ## 파일 이름과 텍스트 레이블 가져오기.
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        ## 프로세서를 통해 이미지 인코딩.
        image = Image.open(os.path.join(self.image_dir,file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        ## 토크나이저를 통해 텍스트 레이블을 인코딩.
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        ## 패딩 토큰은 비용 함수가 무시하게끔 설정하기.
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        ## 변환한 이미지와 텍스트에 대한 인코딩을 딕셔너리에 저장. 모델 입력값으로 쓰일 예정.
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding