#!/usr/bin/env python3
"""
Artisan Marketplace AI Storytelling Backend
Uses Google Cloud Speech API + Google Generative AI (Gemini)
Specialized for generating product descriptions, captions, and hashtags for artisan products
"""

import os
import struct
import tempfile
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import speech
from google.cloud.speech import RecognitionConfig, RecognitionAudio
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

def setup_google_credentials():
    credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
    if credentials_json:
        try:
            # Write the JSON content to a file
            with open('google-credentials.json', 'w') as f:
                f.write(credentials_json)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './google-credentials.json'
            print("✅ Google credentials file created successfully")
        except Exception as e:
            print(f"❌ Failed to create credentials file: {e}")

# Load environment variables from .env file
load_dotenv("../.env")

setup_google_credentials()

# Initialize FastAPI app
app = FastAPI(
    title="Artisan Marketplace AI Storytelling API",
    description="Generate product descriptions, captions, and hashtags using Google APIs",
    version="2.0.0"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://artisan-aiservice-production.up.railway.app",
        "https://ai-powered-marketplace-assistant-fo-seven.vercel.app",  # Add your exact Vercel domain
        "https://ai-powered-marketplace-assistant-for-local-artisans-1fs3pgq22.vercel.app",  # Your other domain from logs
        # Add any other Vercel domains you have
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------- Pydantic models ----------------

class ArtisanStoryRequest(BaseModel):
    language_code: str = "en-US"
    model_name: str = "gemini-1.5-flash"

class ArtisanStoryResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    processing_info: Optional[dict] = None

# ---------------- Audio utilities (from your friend's code) ----------------

def read_wav_header(audio_file_path):
    try:
        with open(audio_file_path, 'rb') as f:
            riff = f.read(4)
            if riff != b'RIFF':
                return None, None
            f.read(4)
            wave = f.read(4)
            if wave != b'WAVE':
                return None, None
            while True:
                chunk_id = f.read(4)
                if not chunk_id:
                    break
                chunk_size = struct.unpack('<I', f.read(4))[0]
                if chunk_id == b'fmt ':
                    audio_format = struct.unpack('<H', f.read(2))[0]
                    num_channels = struct.unpack('<H', f.read(2))[0]
                    sample_rate = struct.unpack('<I', f.read(4))[0]
                    byte_rate = struct.unpack('<I', f.read(4))[0]
                    block_align = struct.unpack('<H', f.read(2))[0]
                    bits_per_sample = struct.unpack('<H', f.read(2))[0]
                    return sample_rate, bits_per_sample
                else:
                    f.seek(chunk_size, 1)
    except Exception as e:
        print(f"⚠️  Could not read WAV header: {e}")
        return None, None

def detect_audio_format(audio_file_path):
    try:
        with open(audio_file_path, 'rb') as f:
            header = f.read(16)
            if header.startswith(b'\x1a\x45\xdf\xa3'):
                return 'webm', speech.RecognitionConfig.AudioEncoding.OGG_OPUS, None
            elif header.startswith(b'OggS'):
                return 'ogg', speech.RecognitionConfig.AudioEncoding.OGG_OPUS, None
            elif header.startswith(b'RIFF') and header[8:12] == b'WAVE':
                sample_rate, bits_per_sample = read_wav_header(audio_file_path)
                if sample_rate:
                    return 'wav', speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate
                else:
                    return 'wav', speech.RecognitionConfig.AudioEncoding.LINEAR16, 16000
            elif header.startswith(b'\xff\xfb') or header.startswith(b'ID3'):
                return 'mp3', speech.RecognitionConfig.AudioEncoding.MP3, 16000
            elif header.startswith(b'\x00\x00\x00\x20ftypM4A'):
                return 'm4a', speech.RecognitionConfig.AudioEncoding.MP3, 16000
            elif header.startswith(b'fLaC'):
                return 'flac', speech.RecognitionConfig.AudioEncoding.FLAC, 16000
            else:
                return 'unknown', speech.RecognitionConfig.AudioEncoding.LINEAR16, 16000
    except Exception as e:
        print(f"⚠️  Header detection failed: {e}")
        return None, None, None

def get_audio_encoding_and_config(audio_file_path):
    actual_format, encoding, sample_rate = detect_audio_format(audio_file_path)
    if actual_format and encoding:
        print(f"🔍 Actual file format detected: {actual_format.upper()}")
        if sample_rate:
            print(f"🔍 Sample rate: {sample_rate} Hz")
        return encoding, sample_rate
    file_extension = audio_file_path.lower().split('.')[-1]
    print(f"⚠️  Using extension-based detection: {file_extension.upper()}")
    if file_extension == 'wav':
        sample_rate, bits_per_sample = read_wav_header(audio_file_path)
        if sample_rate:
            print(f"🔍 WAV file sample rate: {sample_rate} Hz")
            return speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate
        else:
            return speech.RecognitionConfig.AudioEncoding.LINEAR16, 16000
    elif file_extension == 'mp3':
        return speech.RecognitionConfig.AudioEncoding.MP3, 16000
    elif file_extension == 'm4a':
        return speech.RecognitionConfig.AudioEncoding.MP3, 16000
    elif file_extension == 'flac':
        return speech.RecognitionConfig.AudioEncoding.FLAC, 16000
    elif file_extension in ['ogg', 'webm']:
        return speech.RecognitionConfig.AudioEncoding.OGG_OPUS, None
    else:
        return speech.RecognitionConfig.AudioEncoding.LINEAR16, 16000

# ---------------- Core Functions ----------------

def transcribe_audio_with_google(audio_file_path, language_code="en-US"):
    """Transcribe audio using Google Cloud Speech API"""
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        return {"success": False, "error": "GOOGLE_APPLICATION_CREDENTIALS not found in .env file"}
    if not os.path.exists(credentials_path):
        return {"success": False, "error": f"Credentials file not found at: {credentials_path}"}
    if not os.path.exists(audio_file_path):
        return {"success": False, "error": f"Audio file not found: {audio_file_path}"}
    
    try:
        client = speech.SpeechClient()
        with open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()
        
        encoding, sample_rate = get_audio_encoding_and_config(audio_file_path)
        audio = RecognitionAudio(content=content)
        
        if encoding == speech.RecognitionConfig.AudioEncoding.OGG_OPUS:
            config = RecognitionConfig(
                encoding=encoding,
                language_code=language_code,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
            )
        else:
            config = RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
            )
        
        response = client.recognize(config=config, audio=audio)
        
        if response.results:
            transcription = ""
            confidence_scores = []
            for result in response.results:
                if result.alternatives:
                    transcription += result.alternatives[0].transcript + " "
                    confidence_scores.append(result.alternatives[0].confidence)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            return {
                "success": True,
                "transcription": transcription.strip(),
                "confidence": avg_confidence
            }
        else:
            return {
                "success": True,
                "transcription": "",
                "confidence": 0.0
            }
    except Exception as e:
        return {"success": False, "error": f"Error during transcription: {e}"}

def generate_artisan_content_with_gemini(image_base64, user_input="", model_name="gemini-1.5-flash"):
    """Generate artisan product content using Google Gemini"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return {"success": False, "error": "GOOGLE_API_KEY not found in .env file"}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Create specialized prompt for artisan products
        prompt = f"""You are an expert copywriter specializing in handmade and artisan products. 

Analyze this image of a handmade/artisan product and generate content for an online marketplace.

{f"Additional context from the artisan: {user_input}" if user_input else ""}

Please provide EXACTLY in this JSON format:
{{
    "title": "A catchy product title (2-4 words)",
    "description": "A compelling product description (2-3 sentences highlighting craftsmanship, materials, and uniqueness)",
    "caption": "An engaging social media caption (1-2 sentences, friendly and inspiring)",
    "hashtags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"]
}}

Focus on:
- Traditional craftsmanship and techniques
- Quality of materials used
- Cultural significance if applicable
- Uniqueness and handmade nature
- Emotional connection and story
- Use relevant hashtags for artisan/handmade community

Respond ONLY with valid JSON, no additional text."""

        # Prepare content for multimodal generation
        image_data = base64.b64decode(image_base64)
        prompt_parts = [
            prompt,
            {
                "mime_type": "image/png",
                "data": image_data
            }
        ]
        
        response = model.generate_content(prompt_parts)
        
        if response.text:
            # Try to parse JSON response
            try:
                content = json.loads(response.text.strip())
                
                # Validate required fields
                required_fields = ["title", "description", "caption", "hashtags"]
                if all(field in content for field in required_fields):
                    return {"success": True, "content": content}
                else:
                    # Fallback parsing if JSON structure is different
                    return parse_gemini_fallback(response.text.strip(), user_input)
            except json.JSONDecodeError:
                # Fallback to text parsing
                return parse_gemini_fallback(response.text.strip(), user_input)
        else:
            return {"success": False, "error": "Could not generate response from Gemini"}
            
    except Exception as e:
        return {"success": False, "error": f"Error generating content with Gemini: {e}"}

def parse_gemini_fallback(response_text, user_input=""):
    """Fallback parser if JSON format is not followed"""
    try:
        content = {
            "title": "Handcrafted Artisan Product",
            "description": "",
            "caption": "",
            "hashtags": []
        }
        
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract content
            if 'title' in line.lower() or line.startswith('Title:'):
                content["title"] = re.sub(r'^.*?:', '', line).strip().strip('"')
            elif 'description' in line.lower() or line.startswith('Description:'):
                content["description"] = re.sub(r'^.*?:', '', line).strip().strip('"')
            elif 'caption' in line.lower() or line.startswith('Caption:'):
                content["caption"] = re.sub(r'^.*?:', '', line).strip().strip('"')
            elif 'hashtag' in line.lower() or line.startswith('Hashtags:'):
                hashtag_line = re.sub(r'^.*?:', '', line).strip()
                hashtags = [tag.strip('#').strip() for tag in hashtag_line.split() if tag.strip()]
                content["hashtags"] = hashtags[:7]
        
        # Fallback content if parsing failed
        if not content["description"]:
            content["description"] = f"Beautiful handcrafted product showcasing traditional artisan techniques. {user_input if user_input else 'Each piece is unique and made with care by skilled craftspeople.'}"
        
        if not content["caption"]:
            content["caption"] = f"Handmade with love! {user_input if user_input else 'Supporting local artisans and traditional crafts.'}"
            
        if not content["hashtags"]:
            content["hashtags"] = ["handmade", "artisan", "handcrafted", "traditional", "unique", "smallbusiness", "supportlocal"]
        
        return {"success": True, "content": content}
        
    except Exception as e:
        return {"success": False, "error": f"Error in fallback parsing: {e}"}

# ---------------- FastAPI Endpoints ----------------

@app.get("/")
async def root():
    return {
        "message": "Artisan Marketplace AI Storytelling API",
        "version": "2.0.0",
        "powered_by": ["Google Cloud Speech API", "Google Generative AI (Gemini)"],
        "endpoints": {
            "POST /generate-story": "Generate artisan product content from image + optional audio/note",
            "GET /health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    # Check if required environment variables are set
    has_google_credentials = bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    has_google_api_key = bool(os.getenv('GOOGLE_API_KEY'))
    
    return {
        "status": "healthy",
        "google_credentials": has_google_credentials,
        "google_api_key": has_google_api_key,
        "ready": has_google_credentials and has_google_api_key
    }

@app.post("/generate-story", response_model=ArtisanStoryResponse)
async def generate_artisan_story(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    note: Optional[str] = Form(None),
    language_code: str = Form("en-US"),
    model_name: str = Form("gemini-1.5-flash")
):
    try:
        # Validate image upload
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image file")
        
        # Process image
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # Process audio if provided
        user_input = ""
        audio_confidence = 0.0
        
        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.filename.split('.')[-1]}") as temp_file:
                audio_content = await audio.read()
                temp_file.write(audio_content)
                temp_file_path = temp_file.name
            
            try:
                transcription_result = transcribe_audio_with_google(temp_file_path, language_code)
                if transcription_result["success"] and transcription_result.get("transcription"):
                    user_input = transcription_result["transcription"]
                    audio_confidence = transcription_result.get("confidence", 0.0)
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Use note if no audio transcription
        if not user_input and note:
            user_input = note.strip()
        
        # Generate content with Gemini
        content_result = generate_artisan_content_with_gemini(image_base64, user_input, model_name)
        
        if not content_result["success"]:
            return ArtisanStoryResponse(success=False, error=content_result.get("error"))
        
        # Return successful response
        return ArtisanStoryResponse(
            success=True,
            data=content_result["content"],
            processing_info={
                "has_audio_input": bool(audio),
                "has_text_input": bool(note),
                "transcription_confidence": audio_confidence,
                "image_processed": True,
                "model_used": model_name,
                "language_code": language_code
            }
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return ArtisanStoryResponse(
            success=False,
            error=f"Generation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

