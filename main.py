# main.py
import os
import shutil
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

# Third-party heavy libs (import inside functions if you want lazy import)
import easyocr
from pdf2image import convert_from_path
import numpy as np
import google.generativeai as genai

# Load .env (optional on Render — use Env vars)
load_dotenv()

# Logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("certificate-extractor")

# App
app = FastAPI(
    title="Certificate Metadata Extraction API",
    description="Extract metadata from certificate images and PDFs using OCR and Gemini AI",
    version="1.1.0"
)

# CORS (adjust origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_BYTES", 10 * 1024 * 1024))  # default 10MB
POPPLER_PATH = os.getenv("POPPLER_PATH")  # if pdf2image needs it on Render, set this env
EASYOCR_LANGS = os.getenv("EASYOCR_LANGS", "en").split(",")  # e.g. "en" or "en,fr"
EASYOCR_GPU = os.getenv("EASYOCR_GPU", "false").lower() in ("1", "true", "yes")

# Gemini configuration (if key present)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Lazy-loaded resources stored on app state
@app.on_event("startup")
async def startup_event():
    """
    Start up heavy resources. EasyOCR reader is heavy so we initialize it once.
    """
    logger.info("Starting up: initializing OCR reader...")
    try:
        # Initialize reader in threadpool to avoid blocking event loop
        def make_reader():
            return easyocr.Reader(EASYOCR_LANGS, gpu=EASYOCR_GPU)

        app.state.ocr_reader = await run_in_threadpool(make_reader)
        logger.info("OCR reader initialized.")
    except Exception as e:
        logger.exception("Failed to initialize EasyOCR reader on startup.")
        # Keep server up but mark reader missing
        app.state.ocr_reader = None

    # Check Gemini readiness
    app.state.gemini_ready = bool(GEMINI_API_KEY)
    if not app.state.gemini_ready:
        logger.warning("GEMINI_API_KEY not set. Gemini features will be disabled.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down. Cleaning up resources if needed.")
    # Nothing special to cleanup for easyocr, but free memory if you want
    app.state.ocr_reader = None


# Prompt template (kept mostly the same)
PROMPT_TEMPLATE = """
Extract the following fields from the text. Output ONLY in this exact format, one per line:
Student Name - [name or 'not found']
University Name - [name or 'not found']
Degree Name - [degree or 'not found']
Specialization - [spec or 'not found']
Grade - [grade or 'not found']
Certificate Id - [id or 'not found']
Registration Number - [reg or 'not found']
Date of Issue - [date or 'not found']

Text: {text}
"""

# Pydantic response models
class HealthResponse(BaseModel):
    status: str
    ocr_loaded: bool
    gemini_configured: bool

class ExtractResponse(BaseModel):
    status: str
    filename: str
    extracted_text_length: int
    metadata: Dict[str, str]
    saved_json: Optional[str] = None


# Helpers
def allowed_file_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

async def stream_to_tempfile(upload_file: UploadFile, max_size: int) -> str:
    """
    Stream incoming UploadFile to a temporary file while enforcing a size limit.
    Returns the path to the temporary file.
    """
    suffix = Path(upload_file.filename).suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0
    try:
        # read in chunks
        while True:
            chunk = await upload_file.read(1024 * 64)  # 64KB
            if not chunk:
                break
            total += len(chunk)
            if total > max_size:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail=f"File too large. Max allowed size is {max_size} bytes.")
            tmp.write(chunk)
        tmp.flush()
        tmp.close()
        return tmp.name
    finally:
        await upload_file.close()


async def extract_text_from_image(image_path: str) -> str:
    """
    Use EasyOCR to extract text from an image. Run in threadpool because it's CPU-bound.
    """
    reader = getattr(app.state, "ocr_reader", None)
    if reader is None:
        raise HTTPException(status_code=503, detail="OCR reader not available (startup failed).")
    try:
        def do_read():
            # easyocr accepts numpy arrays or image path; numpy array preferred
            img_np = np.array(image_path_to_pil(image_path)) if isinstance(image_path, str) else image_path
            results = reader.readtext(img_np)
            return ' '.join([r[1] for r in results])
        return await run_in_threadpool(do_read)
    except Exception as e:
        logger.exception("OCR extraction from image failed.")
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")


def image_path_to_pil(path: str):
    # Lazy import here
    from PIL import Image
    return Image.open(path).convert("RGB")


async def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convert PDF pages to images and OCR them. Runs conversion and OCR in threadpool.
    """
    reader = getattr(app.state, "ocr_reader", None)
    if reader is None:
        raise HTTPException(status_code=503, detail="OCR reader not available (startup failed).")
    try:
        def do_convert_and_read():
            # convert_from_path may need poppler_path
            pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH) if POPPLER_PATH else convert_from_path(pdf_path)
            texts = []
            for i, page in enumerate(pages):
                logger.info("OCRing PDF page %d/%d", i + 1, len(pages))
                page_np = np.array(page)
                results = reader.readtext(page_np)
                texts.append(' '.join([r[1] for r in results]))
            return "\n\n".join(texts)
        return await run_in_threadpool(do_convert_and_read)
    except Exception as e:
        logger.exception("PDF extraction failed.")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")


def parse_gemini_response_text(raw_text: str) -> Dict[str, str]:
    """
    Parse Gemini's plain text output into standardized fields.
    """
    lines = raw_text.strip().splitlines()
    data: Dict[str, str] = {}
    field_mapping = {
        'student name': 'Student Name',
        'university name': 'University Name',
        'degree name': 'Degree Name',
        'specialization': 'Specialization',
        'grade': 'Grade',
        'certificate id': 'Certificate Id',
        'registration number': 'Registration Number',
        'date of issue': 'Date of Issue'
    }
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # try separators
        if ' - ' in line:
            key, val = line.split(' - ', 1)
        elif ': ' in line:
            key, val = line.split(': ', 1)
        elif '- ' in line:
            key, val = line.split('- ', 1)
        else:
            continue
        key_norm = key.strip().lower()
        value = val.strip()
        std_key = None
        for mk, std in field_mapping.items():
            if mk in key_norm:
                std_key = std
                break
        if std_key is None:
            # fallback: use title-cased key
            std_key = key.strip().title()
        data[std_key] = value
    # ensure all expected present
    for std in field_mapping.values():
        data.setdefault(std, "not found")
    return data


async def extract_metadata_with_gemini(text: str, retries: int = 2) -> Dict[str, str]:
    """
    Call Gemini to extract structured metadata. Retries on transient failures.
    """
    if not getattr(app.state, "gemini_ready", False):
        raise HTTPException(status_code=503, detail="Gemini API not configured (GEMINI_API_KEY missing).")
    prompt = PROMPT_TEMPLATE.format(text=text)
    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            # Use run_in_threadpool if the SDK blocks
            def call_gemini():
                model = genai.GenerativeModel("gemini-2.5-flash")
                resp = model.generate_content(prompt)
                return resp.text if hasattr(resp, "text") else str(resp)
            raw = await run_in_threadpool(call_gemini)
            logger.debug("Gemini raw response: %s", raw[:1000])
            parsed = parse_gemini_response_text(raw)
            return parsed
        except Exception as e:
            logger.exception("Gemini call failed on attempt %d", attempt)
            last_exc = e
            # simple backoff
            if attempt < retries + 1:
                import time
                time.sleep(1 * attempt)
    # out of retries
    raise HTTPException(status_code=502, detail=f"Failed to get response from Gemini: {str(last_exc)}")


def save_json_to_temp(data: Dict, orig_filename: str) -> str:
    """
    Save metadata JSON to a temporary file and return its path.
    """
    safe_name = Path(orig_filename).stem
    tmpdir = tempfile.gettempdir()
    out_path = Path(tmpdir) / f"metadata_{safe_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(out_path)


# Routes
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Certificate Metadata Extraction API", "version": "1.1.0", "endpoints": ["/extract", "/health"]}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        ocr_loaded=bool(getattr(app.state, "ocr_reader", None)),
        gemini_configured=bool(getattr(app.state, "gemini_ready", False))
    )

@app.post("/extract", response_model=ExtractResponse)
async def extract_certificate_metadata(
    request: Request,
    file: UploadFile = File(...),
    save_json: bool = Query(False, description="If true, save metadata JSON to a temp file and return its path")
):
    """
    Extract metadata from a certificate image or PDF.
    - Accepts image or PDF upload.
    - Returns structured metadata JSON.
    """
    # Basic validations
    if not allowed_file_extension(file.filename):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    # optional early content-length check
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE} bytes.")
        except ValueError:
            pass

    tmp_path = None
    try:
        tmp_path = await stream_to_tempfile(file, MAX_FILE_SIZE)
        logger.info("Received file saved to %s", tmp_path)

        ext = Path(file.filename).suffix.lower()
        if ext == ".pdf":
            extracted_text = await extract_text_from_pdf(tmp_path)
        else:
            # for images, pass file path to OCR
            extracted_text = await extract_text_from_image(tmp_path)

        logger.info("Extracted text length: %d", len(extracted_text or ""))

        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="No text could be extracted from the file. Make sure the image is clear/readable.")

        # Gemini metadata extraction
        metadata = await extract_metadata_with_gemini(extracted_text)

        saved_path = None
        if save_json:
            saved_path = save_json_to_temp(metadata, file.filename)

        response_payload = ExtractResponse(
            status="success",
            filename=file.filename,
            extracted_text_length=len(extracted_text),
            metadata=metadata,
            saved_json=saved_path
        )
        return JSONResponse(content=response_payload.dict())
    finally:
        # cleanup
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                logger.exception("Failed to remove temporary file %s", tmp_path)


# Global exception handlers for nicer responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.warning("HTTPException: %s %s", exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"status": "error", "detail": "Internal server error"})

# If run locally for dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
