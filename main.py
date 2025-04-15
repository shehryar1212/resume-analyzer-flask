from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()  # Looks for .env in current directory by default

# Access variables
app = FastAPI()

# Allow frontend requests (for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 Set your Gemini API Key
os.getenv(genai.configure(api_key="GEMINI_API_KEY") ) # Replace with your actual key

model = genai.GenerativeModel("gemini-2.0-flash")

@app.get("/")
def read_root():
    return {"message": "Gemini Resume Analyzer is Live!"}


@app.post("/analyze/")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # ✅ Extract resume text from PDF
        contents = await file.read()
        with fitz.open(stream=contents, filetype="pdf") as doc:
            resume_text = ""
            for page in doc:
                resume_text += page.get_text()

        # ✅ Prepare prompt for Gemini
        prompt = f"""
You are a smart AI resume analyzer. Analyze the following resume and job description.
Return a JSON object with:
- match_score (from 0 to 100)
- missing_keywords (list of what the resume is missing)
- summary (breif to the point suggestion)

Resume:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"
"""

        response = model.generate_content(prompt)
        return JSONResponse(content={"result": response.text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
