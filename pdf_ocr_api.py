from fastapi import FastAPI, UploadFile
import fitz
import easyocr
import tempfile
import numpy as np
import cv2
import os
import re

app = FastAPI()

reader = easyocr.Reader(
    ['en', 'hi']
)

@app.post("/pdf_ocr")

async def pdf_ocr(file: UploadFile):

    try:

        temp = tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".pdf"

        )

        content = await file.read()

        temp.write(content)

        temp.close()

        doc = fitz.open(

            temp.name

        )

        num_pages = len(doc)

        full_text = ""

        confidences = []

        low_conf = 0

        used_ocr = False

        scanned_pages = 0

        digital_pages = 0

        for page in doc:

            native_text = page.get_text(

                "text"

            )

            native_text = native_text.strip()

            if len(native_text) > 100:

                full_text += (

                    native_text +

                    "\n"

                )

                confidences.append(

                    0.99

                )

                digital_pages += 1

                continue

            scanned_pages += 1

            used_ocr = True

            pix = page.get_pixmap(

                dpi=400

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

                image = image[:,:,:3]

            gray = cv2.cvtColor(

                image,

                cv2.COLOR_RGB2GRAY

            )

            _, thresh = cv2.threshold(

                gray,

                0,

                255,

                cv2.THRESH_BINARY +

                cv2.THRESH_OTSU

            )

            result = reader.readtext(

                thresh,

                detail=1

            )

            page_text = "\n".join(

                [

                    r[1]

                    for r in result

                    if len(r)>=2

                ]

            )

            full_text += (

                page_text +

                "\n"

            )

            for r in result:

                if len(r)>=3:

                    conf = float(

                        r[2]

                    )

                    confidences.append(

                        conf

                    )

                    if conf < 0.60:

                        low_conf +=1

        doc.close()

        os.remove(

            temp.name

        )

        full_text = re.sub(

            r'\s+',

            ' ',

            full_text

        ).strip()

        avg_conf = round(

            sum(confidences)

            /

            len(confidences),

            3

        ) if confidences else 0

        if scanned_pages == 0:

            doc_type = "digital_pdf"

        elif digital_pages == 0:

            doc_type = "scanned_pdf"

        else:

            doc_type = "mixed_pdf"

        return {

            "text":

            full_text,

            "pages":

            num_pages,

            "ocr":

            used_ocr,

            "confidence":

            avg_conf,

            "warning_flag":

            avg_conf < 0.80,

            "low_confidence_blocks":

            low_conf,

            "digital_pages":

            digital_pages,

            "scanned_pages":

            scanned_pages,

            "document_type":

            doc_type,

            "engine":

            "pymupdf"

            if scanned_pages==0

            else

            "easyocr+pymupdf"

        }

    except Exception as e:

        return {

            "error":

            str(e)

        }