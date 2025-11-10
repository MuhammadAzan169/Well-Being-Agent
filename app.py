# app.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
from typing import Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing RAG system
from Agent import load_index, BreastCancerRAGSystem, config

app = FastAPI(title="Well Being Agent - Breast Cancer Support")

# Mount static files from current directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Global RAG system instance
rag_system = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    status: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when the app starts"""
    global rag_system
    try:
        print("🚀 Starting Well Being Agent...")
        print("📋 Loading configuration and index...")
        
        index, retriever = load_index()
        if index and retriever:
            rag_system = BreastCancerRAGSystem(index, retriever)
            print("✅ RAG System initialized successfully")
        else:
            print("❌ Failed to initialize RAG system")
            
    except Exception as e:
        print(f"❌ Startup error: {e}")

# Serve the frontend HTML
@app.get("/")
async def serve_frontend():
    return FileResponse('index.html')

# Serve CSS file explicitly
@app.get("/styles.css")
async def serve_css():
    return FileResponse('styles.css', media_type='text/css')

# Serve JS file explicitly
@app.get("/script.js")
async def serve_js():
    return FileResponse('script.js', media_type='application/javascript')

@app.post("/ask-query", response_model=QueryResponse)
async def ask_query(request: QueryRequest):
    """Main endpoint for processing queries"""
    try:
        if not rag_system:
            return QueryResponse(
                answer="System is initializing. Please wait a moment and try again.",
                status="error"
            )
        
        # Process the query using your RAG system
        answer = rag_system.get_enhanced_answer(user_query=request.query)
        
        return QueryResponse(
            answer=answer,
            status="success"
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return QueryResponse(
            answer="Sorry, I encountered an error. Please try again.",
            status="error"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if rag_system else "initializing",
        "rag_system_loaded": rag_system is not None
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🌐 Starting Breast Cancer Support System...")
    print("💬 Backend will be available at: http://localhost:8000")
    print("📱 Open http://localhost:8000 in your browser after startup\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
