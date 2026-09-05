from fastapi import FastAPI, UploadFile
from paddleocr import PaddleOCR
import easyocr
import fitz
import tempfile
import numpy as np
import cv2
import os
import re

app = FastAPI()


####################################
# OCR MODELS
####################################

paddle = PaddleOCR(

    lang='en'

)

easy = easyocr.Reader(

    ['en','hi']

)


####################################
# Paddle OCR
####################################

def paddle_extract(

        image

):

    try:

        result = paddle.predict(

            image

        )

        text = ""

        confidence = []



        for page in result:

            rec = page.get(

                "rec_texts",

                []

            )

            score = page.get(

                "rec_scores",

                []

            )



            text += "\n".join(

                rec

            )



            confidence.extend(

                score

            )



        avg = 0

        if confidence:

            avg = sum(

                confidence

            )/len(

                confidence

            )



        return text,avg



    except Exception:

        return "",0




####################################
# Easy OCR
####################################

def easy_extract(

        image

):

    result = easy.readtext(

        image,

        detail=1

    )



    text = []



    conf = []



    for r in result:

        text.append(

            r[1]

        )



        conf.append(

            float(

                r[2]

            )

        )



    avg = 0



    if conf:

        avg = sum(

            conf

        )/len(

            conf

        )



    return (

        "\n".join(text),

        avg

    )



####################################
# preprocess
####################################

def preprocess(

        image

):


    gray = cv2.cvtColor(

        image,

        cv2.COLOR_RGB2GRAY

    )



    gray = cv2.GaussianBlur(

        gray,

        (3,3),

        0

    )



    thresh = cv2.adaptiveThreshold(

        gray,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        11,

        2

    )



    rgb = cv2.cvtColor(

        thresh,

        cv2.COLOR_GRAY2RGB

    )



    return rgb



####################################
# endpoint
####################################

@app.post(

    "/pdf_ocr"

)

async def pdf_ocr(

        file: UploadFile

):



    temp = tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".pdf"

    )



    content = await file.read()



    temp.write(

        content

    )



    temp.close()



    doc = fitz.open(

        temp.name

    )



    pages = len(

        doc

    )



    digital_pages = 0

    scanned_pages = 0



    confidences = []



    low_conf = 0



    full_text = ""



    engine_used = []



    for page in doc:



        native = page.get_text()



        if len(

            native.strip()

        ) > 100:



            digital_pages += 1



            full_text += native



            full_text += "\n"



            confidences.append(

                1

            )



            continue



        scanned_pages += 1



        pix = page.get_pixmap(

            dpi=250

        )



        image = np.frombuffer(

            pix.samples,

            dtype=np.uint8

        )



        image = image.reshape(

            pix.height,

            pix.width,

            pix.n

        )



        if pix.n == 4:

            image = image[:,:,:3]



        image = preprocess(

            image

        )



        text,conf = paddle_extract(

            image

        )



        engine = "paddle"



        if conf < 0.70:



            text2,conf2 = easy_extract(

                image

            )



            if conf2 > conf:



                text = text2



                conf = conf2



                engine = "easyocr"



        engine_used.append(

            engine

        )



        confidences.append(

            conf

        )



        if conf < 0.60:

            low_conf += 1



        full_text += text

        full_text += "\n"



    doc.close()



    os.remove(

        temp.name

    )



    full_text = re.sub(

        r"\s+",

        " ",

        full_text

    ).strip()



    avg = 0



    if confidences:

        avg = sum(

            confidences

        )/len(

            confidences

        )



    document_type = "digital_pdf"



    if scanned_pages>0:

        document_type = "scanned_pdf"



    if scanned_pages>0 and digital_pages>0:

        document_type = "hybrid_pdf"



    return {



        "text":full_text,



        "pages":pages,



        "digital_pages":

            digital_pages,



        "scanned_pages":

            scanned_pages,



        "document_type":

            document_type,



        "confidence":

            round(

                avg,

                3

            ),



        "warning_flag":

            avg<0.80,



        "ocr":

            scanned_pages>0,



        "low_confidence_blocks":

            low_conf,



        "engine":

            "Hybrid OCR",



        "engines_used":

            list(

                set(

                    engine_used

                )

            )



    }
