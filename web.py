import os
import json
import uvicorn
import shutil
from typing import Dict
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_config
from preprocessing.parser import DocumentParser
from preprocessing.cleaner import TextCleaner
from preprocessing.chunker import DocumentChunker
from retrieval.embedder import TextEmbedder
from retrieval.vector_store import VectorStore
from agents.ner_agent import NERAgent
from agents.sentiment_agent import SentimentAgent
from agents.kpi_agent import KPIAgent
from agents.rag_agent import RAGAgent
from report.report_generator import ReportGenerator

# --- Global Pipeline Components ---
pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize all components once
    config = get_config()
    embedder = TextEmbedder(config)
    pipeline["vector_store"] = VectorStore(config, embedder)
    pipeline["parser"] = DocumentParser()
    pipeline["cleaner"] = TextCleaner()
    pipeline["chunker"] = DocumentChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    pipeline["ner"] = NERAgent(config)
    pipeline["sentiment"] = SentimentAgent(config)
    pipeline["kpi"] = KPIAgent()
    pipeline["rag"] = RAGAgent(config, pipeline["vector_store"])
    pipeline["report_gen"] = ReportGenerator()
    pipeline["current_doc_id"] = ""
    pipeline["current_data"] = {}
    yield
    # Shutdown: Cleanup
    pipeline.clear()

app = FastAPI(title="The Analyst API", lifespan=lifespan)

# Allow the frontend (localhost:5000) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskQuery(BaseModel):
    query: str


@app.post("/api/process")
async def process_document(file: UploadFile = File(...)):
    try:
        # 1. CLEANUP PREVIOUS SESSION DATA
        # Clear the vector store of the previous document if it exists
        if pipeline.get("current_doc_id"):
            try:
                pipeline["vector_store"].delete_document(pipeline["current_doc_id"])
            except Exception as e:
                print(f"Cleanup error (non-critical): {e}")

        # Reset global state
        pipeline["current_data"] = {}
        pipeline["current_doc_id"] = ""

        # 2. PROCEED WITH NEW UPLOAD
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Generate fresh ID and process
        doc_id = f"{file.filename}_{datetime.now().timestamp()}"
        pipeline["current_doc_id"] = doc_id
        
        # ... (rest of your parsing and indexing logic) ...


        doc_id = f"{file.filename}_{datetime.now().timestamp()}"
        pipeline["current_doc_id"] = doc_id

        # Stages 1-7
        parsed_doc = pipeline["parser"].parse(str(temp_path))
        parsed_doc.raw_text = pipeline["cleaner"].clean(parsed_doc.raw_text)
        chunks = pipeline["chunker"].chunk_document(doc_id, parsed_doc.sections, parsed_doc.raw_text)
        pipeline["vector_store"].index_document(doc_id, chunks)
        
        ner_results = pipeline["ner"].extract_from_chunks([c.__dict__ for c in chunks])
        sentiment_results = pipeline["sentiment"].analyze_document([c.__dict__ for c in chunks])
        kpi_results = pipeline["kpi"].extract(parsed_doc.raw_text, parsed_doc.tables)
        
        # Initial RAG Insight
        rag_init = pipeline["rag"].answer("Summary of key risks and highlights", doc_ids=[doc_id])
        
        report = pipeline["report_gen"].generate_report(
            doc_id, parsed_doc, ner_results, sentiment_results, kpi_results, [rag_init]
        )
        
        pipeline["current_data"] = {"report": report}
        os.remove(temp_path)

        return {
            "success": True,
            "document_id": doc_id,
            "company_name": report.company_name,
            "overall_sentiment": report.overall_sentiment,
            "summary": report.summary,
            "kpis": {k: (v[0].raw_match if v else "N/A") for k, v in kpi_results.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: AskQuery):
    if not pipeline.get("current_doc_id"):
        raise HTTPException(status_code=400, detail="No document processed")
    
    res = pipeline["rag"].answer(request.query, doc_ids=[pipeline["current_doc_id"]])
    return {
        "query": request.query,
        "answer": res.answer,
        "chain_of_thought": res.chain_of_thought,
        "evidence_count": len(res.evidence_chunks_used)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)