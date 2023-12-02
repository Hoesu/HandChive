import os
import torch
import warnings
from PIL import Image
from main import load_model
from main import load_processors

def inference(input_dir, processor, model):
    ## input_dir: 인퍼런스 위한 이미지 위치
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## bbox_concat 거친 원본 이미지 line-by-line 조각 리스트 생성.
    lines = os.listdir(input_dir)
    ## 인식한 텍스트 저장할 값
    generated_text = ""
    ## 라인별로 인퍼런스 진행.
    for line in lines:
        image_path = os.path.join(input_dir, line)
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text += processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

warnings.filterwarnings("ignore")
model_path = '/root/HandChive/Text_Recognition/model/trocr.pth'
image_path = '/root/HandChive/Text_Recognition/data/crop_image'

if os.path.exists(model_path):
    print('loading existing model from the designated path...')
    model = load_model(model_path)
    print('loading processors...')
    processor = load_processors()
    print('conducting inference...')
    result = inference(image_path,processor,model)
    print(result)
else:
    print('model does not exist in the designated path.')