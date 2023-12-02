from main import save_model
from main import load_model
from main import create_model
from main import load_processors
from dataset import create_dataloader

import os
import torch
from transformers import AdamW
from datasets import load_metric
from rich.progress import Progress

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer_metric = load_metric("cer")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer

def train(model_path, train_path, num_epochs):
    processor = load_processors()
    print('processor loaded.')
    train_image_path = os.path.join(train_path,'crop_image')
    train_label_path = os.path.join(train_path,'crop_label/crop_label.csv')
    train_dataloader, test_dataloader = create_dataloader(train_image_path,train_label_path,processor)
    print('dataloader created.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")
    model = None
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model(model_path,processor)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    print('optimizer setting complete.')
    
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
    
model_path = '/root/HandChive/Text_Recognition/model/trocr.pth'
train_path = '/root/HandChive/Text_Recognition/data'

train(model_path, train_path, num_epochs=1)