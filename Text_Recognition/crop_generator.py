import os
import json
import pandas as pd
import cv2

def bbox_concat(json_data):
    result = []
    sorted_bboxes = sorted(json_data['Bbox'], key=lambda bbox: min(bbox['y']))

    def process_text(text):
        sorted_text = sorted(text, key=lambda x: x[1])
        line = ' '.join(word[0] for word in sorted_text)
        return line

    def update_bounds(bounds, bbox):
        bounds[0] = min(bounds[0], min(bbox['x']))
        bounds[1] = max(bounds[1], max(bbox['x']))
        bounds[2] = min(bounds[2], min(bbox['y']))
        bounds[3] = max(bounds[3], max(bbox['y']))

    init_bbox = sorted_bboxes[0]
    text = [[init_bbox['data'], min(init_bbox['x'])]]
    bounds = list(init_bbox['x']) + list(init_bbox['y'])

    for bbox in sorted_bboxes[1:]:
        if min(bbox['y']) < (bounds[2] + bounds[3]) / 2 < max(bbox['y']):
            update_bounds(bounds, bbox)
            text.append([bbox['data'], min(bbox['x'])])
        else:
            line = process_text(text)
            result.append([line, bounds])
            
            bounds = list(bbox['x']) + list(bbox['y'])
            text = [[bbox['data'], min(bbox['x'])]]

    line = process_text(text)
    result.append([line, bounds])

    return result

def pre_process(cropped_image):
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    cropped_image = cv2.medianBlur(cropped_image,5)
    cropped_image = cv2.adaptiveThreshold(cropped_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,13,4)
    cropped_image = cv2.fastNlMeansDenoising(cropped_image,None,10,3,10)
    return cropped_image

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

        image_path = os.path.join(label_dir,label_files[i])
        label_path = os.path.join(image_dir,image_files[i])

        with open(label_path, 'r', encoding='UTF-8') as f:
            label = json.load(f)
        
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
                cropped_image = image[bbox[2]:bbox[3],bbox[0]:bbox[1]]
                cropped_image = pre_process(cropped_image)

                id+=1
                name = image_name + '_' + str(id) + '.jpg'
                crop_labels.loc[len(crop_labels.index)] = [name,data]
                
                crop_image_path = os.path.join(crop_image_dir,name)
                cv2.imwrite(crop_image_path, cropped_image)
            
    crop_labels.to_csv(crop_label_dir, index=False, encoding='utf-8')

crop_generate('C:/Users/wjdgh/Desktop/HandChive/Text_Recognition')