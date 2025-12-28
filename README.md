# Retail Analytics Copilot

A local AI agent that answers retail analytics questions by combining RAG (Retrieval-Augmented Generation) over documents and SQL queries over a SQLite database, with DSPy optimization and LangGraph orchestration.

## Overview

This agent processes questions about the Northwind retail database by:
- Retrieving relevant policy/definition documents
- Generating and executing SQL queries
- Synthesizing typed answers with citations
- Self-repairing failed queries (up to 2 attempts)

**Key Features:**
- 100% local execution (no external API calls)
- Typed outputs matching exact format specifications
- Complete citations (both database tables and document chunks)
- Optimized SQL generation using DSPy

## Architecture

### LangGraph Workflow (8 nodes)

1. **Router**: Classifies questions as `rag` (doc-only), `sql` (db-only), or `hybrid` (both)
2. **Retriever**: TF-IDF search over document chunks, returns top-k with scores
3. **Planner**: Extracts constraints from docs (dates, categories, KPI formulas)
4. **NL-to-SQL**: Generates SQLite queries using schema + context
5. **Executor**: Runs SQL safely, captures results and errors
6. **Synthesizer**: Produces typed answers matching format_hint with citations
7. **Repair**: Fixes SQL errors or format mismatches (max 2 iterations)
8. **Validator**: Checks output format compliance

**Conditional Edges:**
- Skip retrieval for SQL-only questions
- Route to repair on SQL failure or format mismatch
- End when validated or max repairs reached

### DSPy Optimization

**Module Optimized:** NL-to-SQL converter

**Method:** BootstrapFewShot with 15 training examples

**Metric:** SQL validity (execution success rate)

**Results:**
- Baseline: 40-60% valid SQL queries
- Optimized: 60-80% valid SQL queries  
- Improvement: +20-40% (exact numbers in `dspy_optimization_results.json`)

The optimizer learns from examples to generate more syntactically correct queries with proper table names, joins, and date handling.

## Assumptions & Trade-offs

1. **CostOfGoods Approximation**: Assumed to be 70% of UnitPrice since Northwind doesn't have a cost field
2. **Date Extraction**: Uses regex patterns to find YYYY-MM-DD dates in documents
3. **Category Matching**: Hardcoded list of 8 Northwind categories for reliable extraction
4. **Repair Strategy**: Simple heuristics (quote table names, retry on error) rather than complex error analysis
5. **Token Limits**: Schema truncated to 500 chars for DSPy prompts to fit Phi-3.5's context window
6. **Confidence Scoring**: Basic heuristic combining retrieval scores and SQL success (not ML-based)

## Setup

### Prerequisites
- Python 3.10+
- 16GB RAM recommended
- ~5GB free disk space

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd retail-analytics-copilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install and start Ollama
# Visit https://ollama.com/download for your OS
ollama serve &

# 4. Download Phi-3.5 model
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# 5. Download Northwind database
mkdir -p data
curl -L -o data/northwind.sqlite \
https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# 6. Create lowercase views
sqlite3 data/northwind.sqlite <<'SQL'
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
SQL
```

### Create Document Corpus

The `docs/` folder contains 4 markdown files that must be created as specified in the assignment:
- `marketing_calendar.md` - Campaign dates
- `kpi_definitions.md` - AOV and Gross Margin formulas
- `catalog.md` - Category list
- `product_policy.md` - Return windows by product type

## Usage

### Run Batch Evaluation

```bash
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```

### Output Format

Each line in `outputs_hybrid.jsonl` follows the contract:

```json
{
  "id": "question_id",
  "final_answer": <matches format_hint type>,
  "sql": "SELECT ... (or empty if RAG-only)",
  "confidence": 0.75,
  "explanation": "Brief 1-2 sentence explanation",
  "citations": [
    "Orders",
    "Order Details", 
    "Products",
    "marketing_calendar::chunk0"
  ]
}
```

### Run DSPy Optimization

```bash
python optimize_nl_to_sql.py
```

This creates 15 training examples, evaluates baseline accuracy, runs BootstrapFewShot, and outputs improvement metrics.

## Project Structure

```
retail-analytics-copilot/
├── agent/
│   ├── graph_hybrid.py                 # LangGraph workflow
│   ├── dspy_signatures.py              # DSPy modules (Router, NL-to-SQL, Synthesizer)
│   ├── rag/
│   │   └── retrieval.py                # TF-IDF document retriever
│   └── tools/
│       └── sqlite_tool.py              # SQLite interface
├── data/
│   └── northwind.sqlite                # Northwind database
├── docs/                               # Document corpus (4 .md files)
├── dspy_optimization_results.jsonl     # optimization results
├── sample_questions_hybrid_eval.jsonl  # Test questions
├── outputs_hybrid.jsonl                # Generated answers
├── run_agent_hybrid.py                 # CLI entrypoint
├── optimize_nl_to_sql.py               # DSPy optimization script
├── validate_outputs.py                 # Validation script
├── requirements.txt
└── README.md
```

## Testing

### Validate Outputs

```bash
python validate_outputs.py
```

Checks:
- Type correctness (int, float, dict, list)
- Citation presence
- SQL execution
- Specific answer validation (e.g., beverages = 14 days)

### Individual Question Testing

```bash
python -m agent.graph_hybrid
```

Runs a single test question through the full pipeline with detailed trace output.

## Evaluation Results

Based on the 6 test questions:
- **Type Correctness**: 4/6 (66.67%)
- **Has Citations**: 3/6 (50%)
- **Has SQL**: 5/6 questions required SQL
- **Confidence**: Average 0.57

## Performance Notes

- **Per-question time**: ~2-5 minutes (mostly LLM inference)
- **Total eval time**: ~20 minutes for 6 questions
- **Model**: Phi-3.5-mini (4-bit quantized, ~2.3GB)
- **Deterministic**: Temperature set to 0.1 for reproducibility

## Limitations

1. **Small LLM constraints**: Phi-3.5-mini sometimes struggles with complex multi-join queries
2. **No streaming**: Waits for full LLM completion (could add streaming for better UX)
3. **Simple repair**: Heuristic-based rather than learning from past errors
4. **No caching**: Re-processes identical queries (could add result cache)
5. **Limited context**: Schema truncated to fit token limits

## Future Improvements

- [-] Add result caching for identical queries
- [ ] Implement smarter repair using error classification
- [ ] Add streaming output for long-running queries
- [ ] Support for more complex KPI formulas
- [ ] Expand training set for DSPy optimization (100+ examples)
- [ ] Add query plan explanation for debugging

## License

MIT

## Contact

Khaled Ehab Attia Hussein
kha.2002.ke@gmail.com
