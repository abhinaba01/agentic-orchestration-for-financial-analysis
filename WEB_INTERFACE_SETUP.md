# The Analyst — Minimalist Financial NLP Interface

A sleek, minimalist web interface for autonomous financial document analysis powered by an agentic orchestration pipeline.

## Features

- **Document Ingestion**: Upload financial documents (PDF, DOCX, TXT) with a single interaction
- **Executive Summary**: Automatically generated insights and key findings
- **KPI Extraction**: Structured metrics with confidence scores
- **Sentiment Analysis**: Financial outlook assessment  
- **Interactive Q&A**: Ask any question about the document using RAG (Retrieval-Augmented Generation)
- **Chain-of-Thought Transparency**: View the agent's reasoning and evidence for each answer
- **Report Export**: Download structured JSON reports

## Visual Design

Inspired by minimalist principles:
- **Color Palette**: Bone white (#F4F4F2) background with charcoal (#1A1A1A) typography
- **Typography**: Playfair Display for headlines, Inter for body text
- **Layout**: Extreme whitespace, subtle 1px dividers, elegant serif emphasis
- **Interactions**: Smooth CSS transitions (0.5s ease) between states
- **No decorations**: Clean, focused interface optimized for financial data presentation

## Tech Stack

### Backend
- **FastAPI**: RESTful API server
- **Python**: Core pipeline (existing agents, parsers, embedders)
  - `NERAgent`: Named entity recognition
  - `SentimentAgent`: Financial sentiment analysis
  - `KPIAgent`: Key performance indicator extraction
  - `RAGAgent`: Retrieval-augmented generation for Q&A

### Frontend
- **Vanilla JavaScript**: No heavy frameworks
- **Tailwind CSS**: Utility-first styling via CDN
- **HTML5**: Semantic markup

## Installation

### 1. Install Dependencies

```bash
# Install CPU-only PyTorch first (recommended on Windows)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment

Create or ensure your `.env` file has:

```env
OPENAI_API_KEY=sk-proj-...
CHROMA_PERSIST_DIR=./chroma_db
ONNX_NER_MODEL_PATH=./models/ner_int8.onnx
ONNX_SENTIMENT_MODEL_PATH=./models/sentiment_int8.onnx
```

### 3. Download Models (Optional)

If you don't have the ONNX models, the system will automatically fallback to HuggingFace models. To use optimized ONNX models, see `training_notebooks/export_to_onnx.ipynb`.

## Running the Web Server

### Start the FastAPI Backend

```bash
# From the project root directory
python web.py
```

The server will start at `http://localhost:8000`

**API Endpoints:**
- `POST /api/process` — Upload and process a document
- `POST /api/ask` — Ask a question about the ingested document
- `GET /api/report/download` — Download the generated report as JSON

### Serve the Frontend

In another terminal, serve `index.html`:

```bash
# Using Python's built-in server
python -m http.server 5000

# Or use any other static server (Live Server, VS Code extension, etc.)
```

Open `http://localhost:5000` in your browser.

## Usage Flow

1. **Open** the web interface at `http://localhost:5000`
2. **Click** "Ingest Document —" to upload a financial document
3. **Wait** for processing to complete (parsing, NER, sentiment, KPI extraction)
4. **Review** the dashboard with:
   - Executive summary
   - Company overview
   - Key performance indicators
   - Market outlook (sentiment gauge)
5. **Ask** questions in the Q&A section
6. **View Logic** toggle to see the agent's chain-of-thought reasoning
7. **Download** the full report as JSON

## API Examples

### Process a Document

```bash
curl -X POST http://localhost:8000/api/process \
  -F "file=@financial_report.pdf"
```

Response:
```json
{
  "success": true,
  "document_id": "financial_report.pdf_1712991234.123",
  "company_name": "Apple Inc.",
  "overall_sentiment": "positive",
  "summary": "...",
  "kpis": {...},
  "key_entities": {...}
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the major revenue drivers?"}'
```

Response:
```json
{
  "query": "What are the major revenue drivers?",
  "answer": "Based on the financial documents, the major revenue drivers are...",
  "chain_of_thought": "Step 1: Retrieved revenue section... Step 2: Analyzed...",
  "evidence_count": 3
}
```

### Download Report

```bash
curl -X GET http://localhost:8000/api/report/download \
  -o financial_report.json
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (index.html)                     │
│         Minimalist UI, Vanilla JS, Tailwind CSS             │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API Calls
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI Backend (web.py)                         │
│  /api/process  │  /api/ask  │  /api/report/download         │
└──────────────────────┬──────────────────────────────────────┘
                       │ Python Pipeline
                       ▼
┌──────────────────────────────────────────────────────────────┐
│           Agentic Orchestration Pipeline                     │
!  ├─ DocumentParser (Stage 1: Parse)                          │
│  ├─ TextCleaner (Stage 2: Clean)                            │
│  ├─ DocumentChunker (Stage 3: Chunk)                        │
│  ├─ TextEmbedder + VectorStore (Stage 4: Embed & Index)    │
│  ├─ NERAgent (Stage 5: Named Entity Recognition)           │
│  ├─ SentimentAgent (Stage 6: Sentiment Analysis)           │
│  ├─ KPIAgent (Stage 7: KPI Extraction)                     │
│  ├─ RAGAgent (Stage 8: Q&A via GPT-4o)                     │
│  └─ ReportGenerator (Stage 9: Report Generation)           │
└──────────────────────────────────────────────────────────────┘
```

## Performance Optimization Tips

1. **Model Caching**: The pipeline initialize agents once at server startup
2. **Document Indexing**: ChromaDB persists embeddings for fast subsequent queries
3. **ONNX Quantization**: Use INT8 quantized models for 4-8x faster inference (onnx_model_path)
4. **Batch Processing**: The backend processes chunks in batches during NER and sentiment analysis
5. **Streaming**: For large documents, consider streaming API responses

## Troubleshooting

### FastAPI Server Won't Start
- Ensure port 8000 is not in use: `netstat -an | grep 8000`
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Frontend Can't Connect to Backend
- Ensure FastAPI is running at `http://localhost:8000`
- Check CORS is enabled in web.py (it is by default)
- Open browser console (F12) for network error details

### Models Not Found
- The system will fallback to HuggingFace models automatically
- For ONNX models, download from the provided links in config.py or see training_notebooks/

### OpenAI API Errors
- Verify `OPENAI_API_KEY` is set in `.env`
- Check your API quota and rate limits
- Ensure the model specified in config (`gpt-4o`) is available in your account

## Customization

### Change Color Palette
Edit the CSS variables in `index.html`:
- Background: `#F4F4F2`
- Text: `#1A1A1A`
- Accent colors in sentiment gauge

### Modify Initial Query
In `web.py`, change the `initial_query` variable in the `/api/process` endpoint.

### Add More Agents
Extend the `/api/process` endpoint to run additional analysis agents and update the dashboard component in `index.html`.

### Export Format
Currently exports JSON. Modify `web.py` `/api/report/download` to support PDF, Markdown, etc.

## Future Enhancements

- [ ] Batch document processing
- [ ] Real-time websocket streaming for long operations
- [ ] PDF export with charts and visualizations
- [ ] Multi-document comparison
- [ ] Saved analysis sessions
- [ ] API authentication (OAuth2)
- [ ] Advanced filtering and drill-down in KPI grid
- [ ] Custom agent orchestration UI

## File Structure

```
NLP_Finance/
├── web.py                         # FastAPI backend server
├── index.html                     # Minimalist frontend UI
├── main.py                        # CLI entry point (existing)
├── config.py                      # Configuration (existing)
├── requirements.txt               # Dependencies (updated)
├── agents/                        # Agent modules (existing)
├── preprocessing/                 # Document processing (existing)
├── retrieval/                     # Vector store and embedding (existing)
├── report/                        # Report generation (existing)
└── output/                        # Generated reports directory
```

## License

Same as the parent project.

## Support

For issues or questions, refer to the main project README or open an issue on GitHub.
