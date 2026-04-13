# The Analyst — Quick Start Guide

## What Is This?

A sleek, minimalist web interface for analyzing financial documents using AI agents. Upload a document, get instant insights, and ask follow-up questions.

## In 5 Minutes

### 1. **Install Dependencies** (Windows)

```powershell
# Install PyTorch (CPU only)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies
pip install -r requirements.txt
```

### 2. **Verify .env Setup**

Make sure your `.env` file has:
```env
OPENAI_API_KEY=sk-proj-your-key-here
```

### 3. **Start the Server**

**Windows:**
```powershell
.\run_web_interface.bat
```

**Mac/Linux:**
```bash
chmod +x run_web_interface.sh
./run_web_interface.sh
```

This will:
- Start FastAPI backend on `http://localhost:8000`
- Start frontend server on `http://localhost:5000`
- Open your browser automatically

### 4. **Use The Interface**

1. Click **"Ingest Document —"**
2. Select a PDF, DOCX, or TXT file
3. Wait for processing
4. Review the dashboard
5. Ask questions about your document
6. Download the report

## Manual Startup (Alternative)

If the startup script doesn't work:

**Terminal 1 - Backend:**
```powershell
python web.py
```

**Terminal 2 - Frontend:**
```powershell
python -m http.server 5000
```

Then open: `http://localhost:5000`

## Testing the API

Once the backend is running, you can test endpoints:

```bash
# Upload a document
curl -X POST http://localhost:8000/api/process \
  -F "file=@your_document.pdf"

# See API docs
# Visit: http://localhost:8000/docs
```

## What Happens Inside

1. **Parse** → Extract text from PDF/DOCX
2. **Clean** → Remove noise and artifacts  
3. **Chunk** → Split into manageable pieces
4. **Embed** → Convert to vectors for retrieval
5. **NER** → Extract entities (companies, people, dates)
6. **Sentiment** → Analyze financial outlook
7. **KPI** → Extract key metrics
8. **RAG** → Answer questions using document context
9. **Report** → Generate structured output

## Keyboard Shortcuts

| Action | Key |
|--------|-----|
| Ask a question | `Enter` (in Q&A input) |
| View reasoning | Click "View Logic" |
| Upload document | Click "Ingest Document —" |

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Port 8000/5000 already in use
Kill the process or change port in `web.py` and `run_web_interface.bat`

### OPENAI_API_KEY error
1. Create `.env` file in project root
2. Add: `OPENAI_API_KEY=sk-proj-...`
3. Restart the server

### Models not downloading
The system will automatically use HuggingFace fallback models. This is normal and slower but works fine.

## Directory Structure

```
NLP_Finance/
├── index.html                     ← Open this in browser (or use startup script)
├── web.py                         ← FastAPI backend
├── run_web_interface.bat          ← Windows startup script
├── run_web_interface.sh           ← Mac/Linux startup script
├── WEB_INTERFACE_SETUP.md         ← Full documentation
├── agents/                        ← NER, Sentiment, KPI, RAG agents
├── preprocessing/                 ← Document parsing & preprocessing
├── retrieval/                     ← Embeddings & vector store
├── report/                        ← Report generation
└── .env                           ← Your API keys (create this)
```

## Next Steps

- **Customize colors** in `index.html` (look for `#F4F4F2` and `#1A1A1A`)
- **Add more initial queries** in `web.py` (search for `initial_query`)
- **Export in other formats** (modify `/api/report/download` in `web.py`)
- **Check full docs** in `WEB_INTERFACE_SETUP.md`

## API Reference

### Process a Document
```
POST /api/process
Content-Type: multipart/form-data
Body: file (PDF, DOCX, or TXT)

Response:
{
  "success": true,
  "document_id": "...",
  "company_name": "...",
  "overall_sentiment": "positive|negative|neutral",
  "summary": "...",
  "kpis": {...},
  "key_entities": {...}
}
```

### Ask a Question
```
POST /api/ask
Content-Type: application/json
Body: {"query": "What are the risks?"}

Response:
{
  "query": "...",
  "answer": "...",
  "chain_of_thought": "...",
  "evidence_count": 3
}
```

### Download Report
```
GET /api/report/download

Returns: JSON file with full analysis
```

### API Docs
```
http://localhost:8000/docs  ← Interactive Swagger documentation
```

## Support

Full documentation: `WEB_INTERFACE_SETUP.md`
