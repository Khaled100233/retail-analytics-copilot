"""
DSPy Signatures and Modules for Retail Analytics Agent
Handles routing, NL-to-SQL conversion, and answer synthesis
"""
import dspy
from typing import Literal, Any
import json
import re


# Configure DSPy to use Ollama
def setup_dspy(model_name: str = "phi3.5:3.8b-mini-instruct-q4_K_M"):
    """Initialize DSPy with Ollama backend"""
    lm = dspy.LM(
        model=f"ollama/{model_name}",
        base_url="http://127.0.0.1:11434",
        max_tokens=500,
        temperature=0.1  # Low temperature for deterministic outputs
    )
    dspy.settings.configure(lm=lm)
    print(f"DSPy configured with model: {model_name}")
    return lm


# ===== ROUTER SIGNATURE =====
class RouteQuestion(dspy.Signature):
    """Classify question into rag, sql, or hybrid route"""
    question = dspy.InputField(desc="The user's question")
    route = dspy.OutputField(
        desc="One of: 'rag' (doc-only), 'sql' (db-only), or 'hybrid' (both needed)"
    )


class Router(dspy.Module):
    """Routes questions to appropriate retrieval strategy"""
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouteQuestion)
    
    def forward(self, question: str) -> str:
        """
        Determine route based on question content
        
        Returns:
            'rag', 'sql', or 'hybrid'
        """
        result = self.classify(question=question)
        route = result.route.lower().strip()
        
        # Normalize to valid routes
        if 'hybrid' in route or 'both' in route:
            return 'hybrid'
        elif 'sql' in route or 'database' in route or 'number' in route:
            return 'sql'
        elif 'rag' in route or 'doc' in route or 'policy' in route:
            return 'rag'
        else:
            # Default to hybrid for safety
            return 'hybrid'


# ===== NL-TO-SQL SIGNATURE =====
class GenerateSQL(dspy.Signature):
    """Convert natural language question to SQL query"""
    question = dspy.InputField(desc="User's question")
    db_schema = dspy.InputField(desc="Database schema with tables and columns")
    context = dspy.InputField(desc="Additional context from documents (dates, categories, etc.)")
    sql_query = dspy.OutputField(desc="Valid SQLite query. Only return the SQL, no explanation.")


class NLToSQL(dspy.Module):
    """Generates SQL queries from natural language"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateSQL)
    
    def forward(self, question: str, schema: str, context: str = "") -> str:
        """
        Generate SQL query
        
        Returns:
            SQL query as string
        """
        result = self.generate(
            question=question,
            db_schema=schema,
            context=context
        )
        
        # Extract SQL from response (handle markdown code blocks)
        sql = result.sql_query.strip()
        
        # Remove markdown code fences if present
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'^```\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)
        
        # Remove any explanatory text after the query
        # Take only the first complete SQL statement
        if ';' in sql:
            sql = sql.split(';')[0] + ';'
        
        return sql.strip()


# ===== SYNTHESIZER SIGNATURE =====
class SynthesizeAnswer(dspy.Signature):
    """Synthesize final answer from SQL results and documents"""
    question = dspy.InputField(desc="Original question")
    format_hint = dspy.InputField(desc="Expected output format (int, float, dict, list)")
    sql_result = dspy.InputField(desc="Results from database query (if any)")
    doc_chunks = dspy.InputField(desc="Relevant document excerpts (if any)")
    final_answer = dspy.OutputField(desc="Answer matching the format_hint exactly. Only return the value, no explanation.")
    explanation = dspy.OutputField(desc="Brief 1-2 sentence explanation")


class Synthesizer(dspy.Module):
    """Produces typed answers with citations"""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)
    
    def forward(self, question: str, format_hint: str, sql_result: Any, doc_chunks: list) -> dict:
        """
        Generate final answer matching format_hint
        
        Returns:
            dict with keys: final_answer, explanation, confidence, citations
        """
        # Format inputs for LLM
        sql_str = json.dumps(sql_result) if sql_result else "None"
        docs_str = "\n".join([f"[{c['id']}] {c['content']}" for c in doc_chunks]) if doc_chunks else "None"
        
        result = self.synthesize(
            question=question,
            format_hint=format_hint,
            sql_result=sql_str,
            doc_chunks=docs_str
        )
        
        # Parse final_answer based on format_hint
        final_answer = self._parse_answer(result.final_answer, format_hint)
        
        # Extract citations
        citations = self._extract_citations(sql_result, doc_chunks)
        
        # Compute confidence (simple heuristic)
        confidence = self._compute_confidence(sql_result, doc_chunks)
        
        return {
            "final_answer": final_answer,
            "explanation": result.explanation,
            "confidence": confidence,
            "citations": citations
        }
    
    def _parse_answer(self, answer_str: str, format_hint: str) -> Any:
        """Parse answer string into correct type based on format_hint"""
        answer_str = answer_str.strip()
        
        try:
            if format_hint == "int":
                # Extract first number
                match = re.search(r'\d+', answer_str)
                return int(match.group()) if match else 0
            
            elif format_hint == "float":
                # Extract first float
                match = re.search(r'\d+\.?\d*', answer_str)
                return round(float(match.group()), 2) if match else 0.0
            
            elif "{" in format_hint or "dict" in format_hint.lower():
                # Try to parse as JSON dict
                # Remove markdown and extra text
                answer_str = re.sub(r'^```json\s*', '', answer_str)
                answer_str = re.sub(r'^```\s*', '', answer_str)
                answer_str = re.sub(r'\s*```$', '', answer_str)
                
                # Find JSON object
                match = re.search(r'\{[^}]+\}', answer_str)
                if match:
                    return json.loads(match.group())
                return {}
            
            elif "list" in format_hint.lower() or "[" in format_hint:
                # Try to parse as JSON list
                answer_str = re.sub(r'^```json\s*', '', answer_str)
                answer_str = re.sub(r'^```\s*', '', answer_str)
                answer_str = re.sub(r'\s*```$', '', answer_str)
                
                # Find JSON array
                match = re.search(r'\[[^\]]+\]', answer_str)
                if match:
                    return json.loads(match.group())
                return []
            
            else:
                return answer_str
        
        except Exception as e:
            print(f"Error parsing answer: {e}")
            # Return sensible defaults
            if "int" in format_hint:
                return 0
            elif "float" in format_hint:
                return 0.0
            elif "list" in format_hint:
                return []
            elif "{" in format_hint:
                return {}
            else:
                return answer_str
    
    def _extract_citations(self, sql_result: Any, doc_chunks: list) -> list:
        """Extract all citations (table names + doc chunk IDs)"""
        citations = []
        
        # Add document chunk citations
        if doc_chunks:
            for chunk in doc_chunks:
                citations.append(chunk['id'])
        
        # Add table citations from SQL result metadata
        if sql_result and isinstance(sql_result, dict):
            if 'tables_used' in sql_result:
                citations.extend(sql_result['tables_used'])
        
        return citations
    
    def _compute_confidence(self, sql_result: Any, doc_chunks: list) -> float:
        """Simple confidence heuristic"""
        confidence = 0.5  # Base confidence
        
        # Increase if we have SQL results
        if sql_result and isinstance(sql_result, dict):
            if sql_result.get('success') and sql_result.get('rows'):
                confidence += 0.3
        
        # Increase if we have doc chunks with good scores
        if doc_chunks:
            avg_score = sum(c.get('score', 0) for c in doc_chunks) / len(doc_chunks)
            confidence += min(0.2, avg_score * 0.2)
        
        return min(1.0, confidence)


# ===== TESTING =====
if __name__ == "__main__":
    print("Testing DSPy Modules...\n")
    
    # Setup
    setup_dspy()
    
    # Test Router
    print("=== Testing Router ===")
    router = Router()
    
    test_questions = [
        "What is the return policy for beverages?",
        "Top 5 products by revenue",
        "Revenue from Beverages during Summer 1997 campaign"
    ]
    
    for q in test_questions:
        route = router.forward(q)
        print(f"Q: {q}")
        print(f"Route: {route}\n")
    
    # Test NL-to-SQL
    print("=== Testing NL-to-SQL ===")
    nl_to_sql = NLToSQL()
    
    simple_schema = """
    Table: Products
    Columns: ProductID, ProductName, UnitPrice
    
    Table: Orders
    Columns: OrderID, OrderDate
    """
    
    sql = nl_to_sql.forward(
        question="Get top 5 most expensive products",
        schema=simple_schema,
        context=""
    )
    print(f"Generated SQL:\n{sql}\n")
    
    # Test Synthesizer
    print("=== Testing Synthesizer ===")
    synthesizer = Synthesizer()
    
    mock_sql_result = {
        "success": True,
        "rows": [{"ProductName": "Chai", "Revenue": 1234.56}],
        "tables_used": ["Products", "Order Details"]
    }
    
    mock_docs = [
        {"id": "product_policy::chunk0", "content": "Beverages unopened: 14 days", "score": 0.8}
    ]
    
    result = synthesizer.forward(
        question="What is the return window for beverages?",
        format_hint="int",
        sql_result=None,
        doc_chunks=mock_docs
    )
    
    print(f"Final Answer: {result['final_answer']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Citations: {result['citations']}")