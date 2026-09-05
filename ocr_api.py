from fastapi import FastAPI, UploadFile, File
import easyocr
import tempfile
import re
import os

app = FastAPI()

reader = easyocr.Reader(['en'])


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "engine": "easyocr"
    }


@app.post("/ocr")

async def ocr(file: UploadFile = File(...)):

    temp = tempfile.NamedTemporaryFile(
        delete=False
    )

    try:

        content = await file.read()

        temp.write(content)

        temp.close()

        result = reader.readtext(temp.name)

        text = "\n".join(
            [r[1] for r in result]
        )

        text = re.sub(
            r'\s+',
            ' ',
            text
        ).strip()

        if len(result) > 0:

            confidence = [
                r[2]
                for r in result
            ]

            avg_conf = round(
                sum(confidence)
                / len(confidence),
                3
            )

        else:

            avg_conf = 0

        return {

            "success": True,

            "filename": file.filename,

            "type": "image",

            "text": text,

            "blocks": len(result),

            "confidence": avg_conf

        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }

    finally:

        if os.path.exists(temp.name):

            os.remove(temp.name)