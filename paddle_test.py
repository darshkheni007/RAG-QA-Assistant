from paddleocr import PaddleOCR
import os

pdf = r"E:\n8n\RAG_Agent\Announcement_Batch_2.pdf"

ocr = PaddleOCR(lang='en')

result = ocr.predict(pdf)

for i,page in enumerate(result):

    print("\n")
    print("="*60)
    print("PAGE",i+1)
    print("="*60)

    for line in page['rec_texts']:

        print(line)