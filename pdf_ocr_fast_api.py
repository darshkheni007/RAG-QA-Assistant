from fastapi import FastAPI, UploadFile, HTTPException
from paddleocr import PaddleOCR

import fitz
import cv2
import numpy as np
import tempfile
import os
import re
import time

app = FastAPI()


mobile_ocr = PaddleOCR(

    lang="en",

    text_detection_model_name="PP-OCRv5_mobile_det",

    text_recognition_model_name="PP-OCRv5_mobile_rec",

    use_doc_orientation_classify=False,

    use_doc_unwarping=False,

    use_textline_orientation=False

)


medium_ocr = PaddleOCR(

    lang="en",

    text_detection_model_name="PP-OCRv5_server_det",

    text_recognition_model_name="PP-OCRv5_server_rec",

    use_doc_orientation_classify=False,

    use_doc_unwarping=False,

    use_textline_orientation=False

)


def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(

        r'([A-Za-z])\1{5,}',

        r'\1',

        text

    )

    return text.strip()


def render_page(page, dpi):

    pix = page.get_pixmap(

        dpi=dpi

    )

    image = np.frombuffer(

        pix.samples,

        dtype=np.uint8

    ).reshape(

        pix.height,

        pix.width,

        pix.n

    )

    if pix.n == 4:

        image = image[:, :, :3]

    elif pix.n == 1:

        image = cv2.cvtColor(

            image,

            cv2.COLOR_GRAY2RGB

        )

    h, w = image.shape[:2]

    MAX_WIDTH = 1400

    if w > MAX_WIDTH:

        scale = MAX_WIDTH / w

        image = cv2.resize(

            image,

            (

                MAX_WIDTH,

                int(

                    h * scale

                )

            )

        )

    return image


def run_ocr(

        model,

        image

):

    result = model.predict(

        image

    )

    texts = []

    scores = []

    if result:

        r = result[0]

        texts = r.get(

            "rec_texts",

            []

        )

        scores = r.get(

            "rec_scores",

            []

        )

    avg = 0

    if scores:

        avg = sum(

            scores

        ) / len(

            scores

        )

    return texts, scores, avg


@app.post(

    "/pdf_fast_ocr"

)

async def pdf_fast_ocr(

        file: UploadFile

):

    total_start = time.time()

    try:

        temp = tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".pdf"

        )

        temp.write(

            await file.read()

        )

        temp.close()

        doc = fitz.open(

            temp.name

        )

        total_pages = len(

            doc

        )

        text_all = []

        confidences = []

        page_times = []

        digital_pages = 0

        scanned_pages = 0

        for i in range(

                total_pages

        ):

            start = time.time()

            page = doc.load_page(

                i

            )

            native = page.get_text()

            native = native.strip()

            if len(

                    native

            ) > 80:

                digital_pages += 1

                text_all.append(

                    native

                )

                confidences.append(

                    1.0

                )

                page_times.append(

                    {

                        "page":

                            i + 1,

                        "mode":

                            "digital",

                        "confidence":

                            1.0,

                        "sec":

                            round(

                                time.time()

                                -

                                start,

                                2

                            )

                    }

                )

                continue

            scanned_pages += 1

            image = render_page(

                page,

                dpi=120

            )

            texts, scores, avg = run_ocr(

                mobile_ocr,

                image

            )

            if avg < 0.88:

                image = render_page(

                    page,

                    dpi=150

                )

                t2, s2, a2 = run_ocr(

                    mobile_ocr,

                    image

                )

                if a2 > avg:

                    texts = t2

                    scores = s2

                    avg = a2

            if avg < 0.83:

                image = render_page(

                    page,

                    dpi=180

                )

                t3, s3, a3 = run_ocr(

                    medium_ocr,

                    image

                )

                if a3 > avg:

                    texts = t3

                    scores = s3

                    avg = a3

            text_all.extend(

                texts

            )

            confidences.extend(

                scores

            )

            page_times.append(

                {

                    "page":

                        i + 1,

                    "mode":

                        "ocr",

                    "confidence":

                        round(

                            avg,

                            3

                        ),

                    "sec":

                        round(

                            time.time()

                            -

                            start,

                            2

                        )

                }

            )

        doc.close()

        os.remove(

            temp.name

        )

        text = "\n".join(

            text_all

        )

        text = clean_text(

            text

        )

        avg_conf = 0

        if confidences:

            avg_conf = round(

                sum(

                    confidences

                )

                /

                len(

                    confidences

                ),

                3

            )

        total = round(

            time.time()

            -

            total_start,

            2

        )

        return {

            "text":

                text,

            "pages":

                total_pages,

            "digital_pages":

                digital_pages,

            "scanned_pages":

                scanned_pages,

            "document_type":

                (

                    "digital_pdf"

                    if scanned_pages == 0

                    else

                    "mixed_pdf"

                ),

            "confidence":

                avg_conf,

            "warning_flag":

                avg_conf < 0.88,

            "ocr":

                scanned_pages > 0,

            "engine":

                "Adaptive PaddleOCR",

            "processing_time":

                total,

            "page_times":

                page_times

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(

                e

            )

        )