import os
import json
import pandas as pd
import cv2

def bbox_concat(json_data):
    
    result = []
    sorted_bboxes = sorted(json_data['Bbox'], key=lambda bbox: min(bbox['y']))
    initBox = sorted_bboxes[0]
    text = [[initBox['data'], min(initBox['x'])]]
    minX, maxX, minY, maxY = min(initBox['x']), max(initBox['x']), min(initBox['y']), max(initBox['y'])
    midLine = (minY + maxY) / 2

    for bbox in sorted_bboxes[1:]:
        if min(bbox['y']) < midLine < max(bbox['y']):
            minX = min(minX, min(bbox['x']))
            maxX = max(maxX, max(bbox['x']))
            minY = min(minY, min(bbox['y']))
            maxY = max(maxY, max(bbox['y']))
            text.append([bbox['data'], min(bbox['x'])])
        else:
            text = sorted(text, key=lambda x: x[1])
            line = ' '.join(word[0] for word in text)
            result.append([line,[minX, maxX, minY, maxY]])
        
            minX, maxX, minY, maxY = min(bbox['x']), max(bbox['x']), min(bbox['y']), max(bbox['y'])
            midLine = (minY + maxY) / 2
            text = [[bbox['data'],min(bbox['x'])]]
        
    text = sorted(text, key=lambda x: x[1])
    line = ' '.join(word[0] for word in text)
    result.append([line,[minX, maxX, minY, maxY]])

    return result

def pre_process(cropped_image):
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image = cv2.medianBlur(cropped_image,5)
    cropped_image = cv2.adaptiveThreshold(cropped_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,13,4)
    cropped_image = cv2.fastNlMeansDenoising(cropped_image,None,10,3,10)
    return cropped_image

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def crop_generate(root_dir):
    
    label_dir = os.path.join(root_dir,'data/label')
    image_dir = os.path.join(root_dir,'data/image')
    crop_label_dir = os.path.join(root_dir,'data/crop_label/crop_label.csv')
    crop_image_dir = os.path.join(root_dir,'data/crop_image')

    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    crop_labels = pd.DataFrame(columns=['file_name','text'])
    n_iter = len(image_files)

    for i in range(n_iter):

        label_path = os.path.join(label_dir,label_files[i])
        image_path = os.path.join(image_dir,image_files[i])
        
        label = load_json_file(label_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_name = label['Images']['identifier']
        
        bbox_sorted = bbox_concat(label)
        id = 0

        if image is None:
            print("Error: Image not loaded.")
        else:
            for item in bbox_sorted:
                data = item[0]
                bbox = item[1]
                print([bbox[2],bbox[3],bbox[0],bbox[1]])
                cropped_image = image[bbox[2]:bbox[3],bbox[0]:bbox[1]]
                cropped_image = pre_process(cropped_image)

                id+=1
                name = image_name + '_' + str(id) + '.jpg'
                crop_labels.loc[len(crop_labels.index)] = [name,data]
                
                crop_image_path = os.path.join(crop_image_dir,name)
                cv2.imwrite(crop_image_path, cropped_image)
            
    crop_labels.to_csv(crop_label_dir, index=False, encoding='utf-8')

crop_generate('/root/HandChive/Text_Recognition')