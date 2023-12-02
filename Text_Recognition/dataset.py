import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def create_dataloader(image_dir, label_dir, processor):
    df = pd.read_csv(label_dir, encoding='utf-8')
    ## 데이터 스플릿
    train_df, test_df = train_test_split(df, test_size=0.2, random_state = 0)
    ## 인덱스 초기화
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    ## 데이터셋 클래스에 먹여주기
    train_dataset = HandWriting(image_dir=image_dir, df=train_df, processor=processor)
    test_dataset = HandWriting(image_dir=image_dir, df=test_df, processor=processor)
    ## 스크립트 출력
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(test_dataset))
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