"""
FastAPI web server for the Financial NLP Agentic Pipeline.
Exposes REST endpoints for document ingestion, processing, and Q&A.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Financial NLP Agentic Pipeline",
    description="Autonomous financial document analysis and orchestration",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State (Pipeline Instances)
# ============================================================================

config = None
parser_stage = None
cleaner = None
chunker = None
embedder = None
vector_store = None
ner_agent = None
sentiment_agent = None
kpi_agent = None
rag_agent = None
report_generator = None

# Store the last processed document in memory for the session
current_document_data: Dict = {}
current_document_id: str = ""


def initialize_pipeline():
    """Initialize all pipeline components."""
    global config, parser_stage, cleaner, chunker, embedder, vector_store
    global ner_agent, sentiment_agent, kpi_agent, rag_agent, report_generator

    config = get_config()
    parser_stage = DocumentParser()
    cleaner = TextCleaner()
    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    embedder = TextEmbedder(config)
    vector_store = VectorStore(config, embedder)
    ner_agent = NERAgent(config)
    sentiment_agent = SentimentAgent(config)
    kpi_agent = KPIAgent()
    rag_agent = RAGAgent(config, vector_store)
    report_generator = ReportGenerator()


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on server startup."""
    initialize_pipeline()


# ============================================================================
# Pydantic Models
# ============================================================================

class AskQuery(BaseModel):
    """Request model for Q&A queries."""
    query: str


class ProcessResponse(BaseModel):
    """Response model for document processing."""
    success: bool
    document_id: str
    company_name: str
    overall_sentiment: str
    summary: str
    kpis: Dict
    key_entities: Dict
    message: str = ""


class AskResponse(BaseModel):
    """Response model for Q&A queries."""
    query: str
    answer: str
    chain_of_thought: str
    evidence_count: int


# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Financial NLP Pipeline",
        "version": "1.0.0"
    }


@app.post("/api/process", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Process a financial document.
    
    - Parses the document
    - Runs NER, Sentiment, and KPI agents
    - Stores results in memory
    - Returns dashboard data
    """
    global current_document_data, current_document_id

    try:
        # Save uploaded file to temporary location
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename

        with open(temp_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Generate document ID
        current_document_id = f"{file.filename}_{datetime.now().timestamp()}"

        # Stage 1: Parse
        parsed_doc = parser_stage.parse(str(temp_path))

        # Stage 2: Clean
        cleaned_text = cleaner.clean(parsed_doc.full_text)

        # Stage 3: Chunk
        chunks = chunker.chunk(cleaned_text)

        # Stage 4: Embed and index
        vector_store.add_documents(
            doc_id=current_document_id,
            chunks=chunks,
            doc_source=file.filename
        )

        # Stage 5: NER Agent
        ner_results = []
        for chunk in chunks[:5]:  # Process first 5 chunks for NER
            chunk_ner = ner_agent.extract_entities(chunk)
            ner_results.extend(chunk_ner)

        # Stage 6: Sentiment Agent
        sentiment_results = {}
        for chunk in chunks[:10]:  # Process first 10 chunks for sentiment
            sent_result = sentiment_agent.analyze_chunk(chunk)
            sentiment_results[sent_result.chunk_id] = sent_result

        # Stage 7: KPI Agent
        kpi_results = kpi_agent.extract_kpis(cleaned_text)

        # Stage 8: RAG Agent (initial query)
        initial_query = "What are the key financial highlights and risks?"
        rag_result = rag_agent.answer(initial_query, doc_ids=[current_document_id])

        # Stage 9: Generate Report
        financial_report = report_generator.generate_report(
            doc_id=current_document_id,
            parsed_doc=parsed_doc,
            ner_results=ner_results,
            sentiment_results=sentiment_results,
            kpi_results=kpi_results,
            rag_answers=[rag_result]
        )

        # Store in memory for subsequent queries
        current_document_data = {
            "report": financial_report,
            "ner_results": ner_results,
            "sentiment_results": sentiment_results,
            "kpi_results": kpi_results,
            "chunks": chunks,
            "parsed_doc": parsed_doc,
        }

        # Clean up temp file
        os.remove(temp_path)

        return ProcessResponse(
            success=True,
            document_id=current_document_id,
            company_name=financial_report.company_name,
            overall_sentiment=financial_report.overall_sentiment,
            summary=financial_report.summary,
            kpis=financial_report.kpis,
            key_entities=financial_report.key_entities,
            message="Document processed successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskQuery):
    """
    Answer a question about the ingested document.
    Uses RAG with the document context.
    """
    global current_document_data, current_document_id

    if not current_document_data:
        raise HTTPException(
            status_code=400,
            detail="No document ingested. Please process a document first."
        )

    try:
        # Query the RAG agent
        rag_result = rag_agent.answer(
            request.query,
            doc_ids=[current_document_id],
            top_k=5
        )

        return AskResponse(
            query=request.query,
            answer=rag_result.answer,
            chain_of_thought=rag_result.chain_of_thought,
            evidence_count=len(rag_result.evidence_chunks_used)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )


@app.get("/api/report/download")
async def download_report():
    """
    Download the generated report as JSON.
    """
    global current_document_data, current_document_id

    if not current_document_data:
        raise HTTPException(
            status_code=400,
            detail="No document processed. Please process a document first."
        )

    try:
        report = current_document_data["report"]
        
        # Convert dataclass to dict
        report_dict = {
            "doc_id": report.doc_id,
            "company_name": report.company_name,
            "report_date": report.report_date,
            "overall_sentiment": report.overall_sentiment,
            "sentiment_breakdown": report.sentiment_breakdown,
            "key_entities": report.key_entities,
            "kpis": report.kpis,
            "rag_insights": [
                {
                    "query": insight.query,
                    "answer": insight.answer,
                } for insight in report.rag_insights
            ],
            "summary": report.summary,
            "generated_at": report.generated_at,
        }

        # Write to temp JSON file
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / f"{current_document_id}_report.json"

        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        return FileResponse(
            report_path,
            filename=f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            media_type="application/json"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
