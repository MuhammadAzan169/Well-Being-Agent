# app.py - Enhanced with voice query support (Text responses only)
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os
import asyncio
import time
import uuid
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path to fix import issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Global RAG system instance
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    global rag_system
    try:
        logger.info("🚀 Starting Well Being Agent...")
        
        # Import refactored Agent module (with safety, citations, improved RAG)
        try:
            from Agent_v2 import rag_system as imported_rag
            rag_system = imported_rag
            logger.info("✅ Loaded refactored Agent_v2 module")
        except ImportError:
            from Agent import load_index, BreastCancerRAGSystem
            logger.info("📋 Loading from legacy Agent module...")
            await asyncio.sleep(2)
            index, retriever = load_index()
            if index and retriever:
                rag_system = BreastCancerRAGSystem(index, retriever)
            else:
                rag_system = None
        
        if rag_system:
            logger.info("✅ RAG System initialized successfully")
        else:
            logger.error("❌ Failed to initialize RAG system")        
        # Start background cleanup task
        asyncio.create_task(cleanup_old_audio_files())            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Well Being Agent...")

app = FastAPI(
    title="Well Being Agent - Breast Cancer Support",
    description="AI-powered breast cancer support system providing evidence-based information and emotional support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    language: str = "auto"
    response_type: str = "text"

class QueryResponse(BaseModel):
    answer: str
    status: str
    language: str = "english"
    sources: list = []

class VoiceResponse(BaseModel):
    text: str
    language: str = "english"
    status: str = "success"

# Create directories if they don't exist
os.makedirs("static/audio", exist_ok=True)
logger.info(f"📁 Created directory structure: static/audio")
logger.info(f"📁 Current working directory: {os.getcwd()}")

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted successfully")
except Exception as e:
    logger.error(f"❌ Failed to mount static files: {e}")

# Frontend serving
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    try:
        if not os.path.exists('index.html'):
            logger.error("❌ index.html not found!")
            fallback_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Well Being Agent - System Running</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
                    .status { color: green; font-weight: bold; }
                    .error { color: red; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 Well Being Agent - Backend Running</h1>
                    <p class="status">✅ Server is running successfully</p>
                    <p class="error">⚠️ index.html file not found</p>
                    <p>Current directory: """ + os.getcwd() + """</p>
                    <p>Static audio directory: """ + str(os.path.exists('static/audio')) + """</p>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=fallback_html, status_code=200)
        
        return FileResponse('index.html')
        
    except Exception as e:
        logger.error(f"❌ Error serving frontend: {e}")
        return JSONResponse(
            {"error": "Frontend serving failed", "details": str(e)}, 
            status_code=500
        )

@app.get("/styles.css")
async def serve_css():
    """Serve CSS file"""
    try:
        if os.path.exists('styles.css'):
            return FileResponse('styles.css', media_type='text/css')
        else:
            return JSONResponse({"error": "CSS file not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": "CSS serving failed"}, status_code=500)

@app.get("/script.js")
async def serve_js():
    """Serve JavaScript file"""
    try:
        if os.path.exists('script.js'):
            return FileResponse('script.js', media_type='application/javascript')
        else:
            return JSONResponse({"error": "JavaScript file not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": "JavaScript serving failed"}, status_code=500)

@app.post("/ask-query", response_model=QueryResponse)
async def ask_query(request: QueryRequest):
    """Main endpoint for processing queries"""
    try:
        if not rag_system:
            return QueryResponse(
                answer="I'm currently initializing. Please wait a moment and try again.",
                status="error",
                language="english"
            )
        
        if not request.query or not request.query.strip():
            return QueryResponse(
                answer="Please enter a question about breast cancer support.",
                status="error",
                language="english"
            )
        
        # Determine language
        if request.language == "auto":
            detected_language = rag_system.detect_language(request.query)
        else:
            detected_language = request.language
        
        logger.info(f"🌐 Processing query in {detected_language}, Type: {request.response_type}")
        
        # Use enhanced method with sources for citation support
        if hasattr(rag_system, 'get_enhanced_answer_with_sources'):
            result = rag_system.get_enhanced_answer_with_sources(
                user_query=request.query,
                language=detected_language,
                response_type=request.response_type
            )
            return QueryResponse(
                answer=result["answer"],
                status="success",
                language=result.get("language", detected_language),
                sources=result.get("sources", [])
            )
        else:
            # Fallback for legacy Agent module
            answer = rag_system.get_enhanced_answer(
                user_query=request.query,
                language=detected_language,
                response_type=request.response_type
            )
            return QueryResponse(
                answer=answer,
                status="success",
                language=detected_language
            )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            answer="I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
            status="error",
            language="english"
        )

@app.post("/voice-query", response_model=VoiceResponse)
async def process_voice_query(
    file: UploadFile = File(...),
    language: str = "auto"  # Auto-detect language from speech
):
    """Process voice query and return TEXT response only (English & Urdu)"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        logger.info(f"🎤 Processing voice query - Language preference: {language}")
        
        # Import audio processor with proper error handling
        try:
            from audio_processor import audio_processor
        except ImportError as e:
            logger.error(f"❌ Failed to import audio_processor: {e}")
            return VoiceResponse(
                text="Audio processing service is currently unavailable.",
                status="error",
                language="english"
            )
        
        # Convert speech to text with language detection
        stt_result = await audio_processor.speech_to_text(file, language)
        
        if not stt_result or not stt_result.get('text'):
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        query_text = stt_result['text']
        detected_language = stt_result.get('language', 'english')
        
        logger.info(f"📝 Transcribed text ({detected_language}): {query_text}")
        
        # Process the query through RAG system
        if not rag_system:
            return VoiceResponse(
                text="System is initializing. Please try again in a moment.",
                status="error",
                language=detected_language
            )
        
        # ✅ Always use TEXT response type for voice queries
        answer = rag_system.get_enhanced_answer(
            user_query=query_text,
            language=detected_language,  # Use detected language
            response_type="text"  # Always text response
        )
        
        logger.info(f"✅ Voice query processed successfully - Response in {detected_language}")
        
        return VoiceResponse(
            text=answer,  # Always return text
            language=detected_language,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice query: {e}")
        return VoiceResponse(
            text="Sorry, I encountered an error processing your voice message.",
            status="error",
            language="english"
        )

# Audio serving endpoint (kept for any future use)
@app.get("/audio/{filename}")
async def serve_audio_direct(filename: str):
    """Direct audio serving endpoint"""
    try:
        audio_path = os.path.join("static", "audio", filename)
        logger.info(f"🔍 Direct audio request for: {filename}")
        
        if not os.path.exists(audio_path):
            logger.error(f"❌ Audio file not found: {audio_path}")
            raise HTTPException(status_code=404, detail=f"Audio file {filename} not found")
        
        # Determine content type
        if filename.endswith('.mp3'):
            media_type = "audio/mpeg"
        elif filename.endswith('.wav'):
            media_type = "audio/wav"
        else:
            media_type = "audio/mpeg"
        
        logger.info(f"🔊 Serving audio file: {audio_path}")
        return FileResponse(audio_path, media_type=media_type, filename=filename)
        
    except Exception as e:
        logger.error(f"❌ Error serving audio file: {e}")
        raise HTTPException(status_code=500, detail="Error serving audio file")

@app.get("/debug-audio")
async def debug_audio():
    """Debug endpoint to check audio file locations"""
    import glob
    
    audio_info = {
        "current_directory": os.getcwd(),
        "static_directory_exists": os.path.exists("static"),
        "static_audio_exists": os.path.exists("static/audio"),
        "audio_files_in_static_audio": [],
        "static_files_mounted": True
    }
    
    # Check static/audio directory
    if os.path.exists("static/audio"):
        audio_files = glob.glob("static/audio/*.mp3") + glob.glob("static/audio/*.wav")
        audio_info["audio_files_in_static_audio"] = [
            {
                "name": os.path.basename(f),
                "size": os.path.getsize(f),
                "path": f,
                "absolute_path": os.path.abspath(f),
            }
            for f in audio_files
        ]
    
    return JSONResponse(audio_info)

@app.get("/predefined-questions")
async def get_predefined_questions(language: str = "english"):
    """Get predefined questions for breast cancer patients"""
    try:
        if not rag_system:
            return JSONResponse({
                "questions": [],
                "status": "system_initializing"
            })
        
        questions = rag_system.get_predefined_questions(language)
        return JSONResponse({
            "questions": questions,
            "status": "success",
            "language": language
        })
        
    except Exception as e:
        logger.error(f"Error getting predefined questions: {e}")
        return JSONResponse({
            "questions": [],
            "status": "error"
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy" if rag_system else "initializing",
        "rag_system_loaded": rag_system is not None,
        "service": "Well Being Agent - Breast Cancer Support",
        "version": "1.0.0"
    }
    
    return JSONResponse(health_status)

@app.get("/info")
async def system_info():
    """System information endpoint"""
    info = {
        "name": "Well Being Agent",
        "description": "AI-powered breast cancer support system",
        "version": "1.0.0",
        "status": "ready" if rag_system else "initializing",
        "features": [
            "Breast cancer information",
            "Treatment guidance", 
            "Fertility options",
            "Recovery timelines",
            "Emotional support",
            "Multilingual support (English/Urdu)",
            "Voice query support (Text responses)"
        ]
    }
    return JSONResponse(info)

# Debug endpoint to check file existence
@app.get("/debug-files")
async def debug_files():
    """Check if required files exist"""
    files = {
        'index.html': os.path.exists('index.html'),
        'styles.css': os.path.exists('styles.css'),
        'script.js': os.path.exists('script.js'),
        'Agent.py': os.path.exists('Agent.py'),
        'audio_processor.py': os.path.exists('audio_processor.py'),
        'current_directory': os.getcwd()
    }
    return JSONResponse(files)

async def cleanup_old_audio_files():
    """Clean up audio files older than 1 hour"""
    while True:
        try:
            audio_dir = os.path.join("static", "audio")
            if os.path.exists(audio_dir):
                current_time = time.time()
                for filename in os.listdir(audio_dir):
                    file_path = os.path.join(audio_dir, filename)
                    if os.path.isfile(file_path):
                        # Delete files older than 1 hour
                        if current_time - os.path.getctime(file_path) > 3600:
                            os.remove(file_path)
                            logger.info(f"🧹 Cleaned up old audio file: {filename}")
        except Exception as e:
            logger.error(f"Error cleaning up audio files: {e}")
        
        await asyncio.sleep(3600)

# Note: cleanup_old_audio_files runs inside lifespan, not via @app.on_event

# Fallback route for SPA
@app.get("/{full_path:path}")
async def serve_frontend_fallback(full_path: str):
    """Fallback to serve index.html for SPA routing"""
    if os.path.exists(full_path) and full_path != "":
        return FileResponse(full_path)
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    # Use port 7860 for Hugging Face, 8000 for local development
    port = int(os.environ.get("PORT", 7860))  # CHANGED: Default to 7860
    logger.info(f"🌐 Starting Well Being Agent Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")