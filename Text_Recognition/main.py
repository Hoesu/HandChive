import os
import pandas as pd
from PIL import Image
from rich.progress import Progress

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_metric

from tqdm.notebook import tqdm
import torchvision.models as models
from sklearn.model_selection import train_test_split
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, DeiTImageProcessor, AdamW


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

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer_metric = load_metric("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer

def train(model, train_dataloader, test_dataloader, device, num_epochs=1):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=len(train_dataloader))
            for batch in train_dataloader:
                # Move batch to device
                for k, v in batch.items():
                    batch[k] = v.to(device)

                # Forward + Backward + Optimize
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                progress.update(task, advance=1)

        print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))

        # Evaluate
        model.eval()
        valid_cer = 0.0
        with Progress() as progress:
            task = progress.add_task("[green]Evaluating...", total=len(test_dataloader))
            with torch.no_grad():
                for batch in test_dataloader:
                    # Run batch generation
                    outputs = model.generate(batch["pixel_values"].to(device))
                    # Compute metrics
                    cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                    valid_cer += cer
                    progress.update(task, advance=1)

        print("Validation CER:", valid_cer / len(test_dataloader))

    # Save the trained model
    save_model(model, '/root/HandChive/Text_Recognition/model/trocr.pth')

def inference(input_dir, processor, model, device):
    lines = os.listdir(input_dir)
    generated_text = ""
    for line in lines:
        image_path = os.path.join(input_dir, line)
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text += processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


processor = load_processors()
#create_model('/root/HandChive/Text_Recognition/model/trocr.pth')
model = load_model('/root/HandChive/Text_Recognition/model/trocr.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#train_dataloader, test_dataloader = dataloader('/root/HandChive/Text_Recognition/data/crop_image','/root/HandChive/Text_Recognition/data/crop_label/crop_label.csv',processor)
#train(model, train_dataloader, test_dataloader, device)
inference('/root/HandChive/Text_Recognition/data/crop_image',processor,model,device)