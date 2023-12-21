import os
import cv2
import json
import torch
import easyocr
import warnings
import pandas as pd
from PIL import Image
from main import load_model
from main import load_processors

def bbox_concat(bbox_list):
    ##  정렬 결과 저장할 빈 리스트 생성.
    result = []
    ## 입력값으로 받은 리스트를 Y_max 순으로 정렬
    sorted_bboxes = sorted(bbox_list, key=lambda x: x[0][3][1])
    ## 첫번째 박스 받아오기
    initBox = sorted_bboxes[0]
    ## X, Y 좌표 최소값 최대값 초기화.
    minX, maxX, minY, maxY = int(initBox[0][0][0]), int(initBox[0][1][0]), int(initBox[0][0][1]), int(initBox[0][2][1])
    midLine = (minY + maxY) / 2

    for bbox in sorted_bboxes[1:]:
        if bbox[0][0][1] <= midLine <= bbox[0][2][1]:
            minX = min(minX, int(bbox[0][0][0]))
            maxX = max(maxX, int(bbox[0][1][0]))
            minY = min(minY, int(bbox[0][0][1]))
            maxY = max(maxY, int(bbox[0][2][1]))
        else:
            result.append([minX, maxX, minY, maxY])
            minX, maxX, minY, maxY = int(bbox[0][0][0]), int(bbox[0][1][0]), int(bbox[0][0][1]), int(bbox[0][2][1])
            midLine = (minY + maxY) / 2
            
    result.append([minX, maxX, minY, maxY])        
    return result

def pre_process(cropped_image):
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #cropped_image = cv2.medianBlur(cropped_image,5)
    #cropped_image = cv2.adaptiveThreshold(cropped_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    #cv2.THRESH_BINARY,13,4)
    #cropped_image = cv2.fastNlMeansDenoising(cropped_image,None,10,3,10)
    return cropped_image
            
def crop_generate(image_path, save_path, bbox_list):
    
    bbox_list = bbox_concat(bbox_list)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    id = 0

    if image is None:
        print("Error: Image not loaded.")
    else:
        for bbox in bbox_list:
            cropped_image = image[bbox[2]:bbox[3],bbox[0]:bbox[1]]
            cropped_image = pre_process(cropped_image)

            id+=1
            name = 'line_' + str(id) + '.jpg'
                
            crop_image_path = os.path.join(save_path, name)
            cv2.imwrite(crop_image_path, cropped_image)
            
def inference(cropped_image_path, processor, model):
    ## cropped_image_path: 인퍼런스 위한 이미지 위치
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## bbox_concat 거친 원본 이미지 line-by-line 조각 리스트 생성.
    lines = os.listdir(cropped_image_path)
    ## 인식한 텍스트 저장할 값
    generated_text = ""
    ## 라인별로 인퍼런스 진행.
    for line in lines:
        image_path = os.path.join(cropped_image_path, line)
        image = Image.open(image_path).convert('RGB')
        pixel_values = processor(image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text += processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text += '/n'
    return generated_text