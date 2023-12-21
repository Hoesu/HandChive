import os
import torch
import easyocr
import warnings
from PIL import Image
from main import load_model
from main import load_processors
from inference import inference
from inference import crop_generate

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib import colors

warnings.filterwarnings("ignore")
original_image_path = '/root/HandChive/pdf_converter/inference/original_image'
cropped_image_path = '/root/HandChive/pdf_converter/inference/cropped_image'
pdf_path = '/root/HandChive/pdf_converter/inference/pdf'
model_path = '/root/HandChive/pdf_converter/model/trocr.pth'

if os.path.exists(model_path):
    print('loading existing model from the designated path...')
    model = load_model(model_path)
    print('loading processors...')
    processor = load_processors()
else:
    print('model does not exist in the designated path.')

output_text = ''
reader = easyocr.Reader(['ko','en'])
names = os.listdir(original_image_path)

for name in names:
    image = os.path.join(original_image_path, name)
    bbox_list = reader.readtext(image)
    crop_generate(image, cropped_image_path, bbox_list)
    output_text += inference(cropped_image_path, processor, model)

output_text_list = output_text.split('/n')
pdfmetrics.registerFont(TTFont("맑은고딕", "malgun.ttf"))

doc = SimpleDocTemplate("result.pdf", pagesize=letter,
        rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
                    name="default", 
                    fontName="맑은고딕", 
                    fontSize=12,
                    alignment=TA_LEFT,
                    textColor=colors.black,
                    borderPadding=0,
                    rightIndent=0,
                    leftIndent=0,
                    spaceAfter=0,
                    spaceBefore=0,
                    splitLongWords=True,
                    spaceShrinkage=0.05,
                    wordWrap="LTR"
                ))

flowables = []
for line in output_text_list:
    if '/b' in line:
        line=line.replace('/b', '<b>')
        line=line.replace('/', '</b>')
    if '/i' in line:
        line=line.replace('/i', '<i>')
        line=line.replace('/', '</i>')
    if '/u' in line:
        line=line.replace('/u', '<u>')
        line=line.replace('/', '</u>')
    if '/r' in line:
        line=line.replace('/r', '<font face="맑은고딕" color="red">')
        line=line.replace('/', '</font>')
    para = Paragraph(line, style=styles["default"])
    print(line)
    flowables.append(para)

doc.build(flowables)
