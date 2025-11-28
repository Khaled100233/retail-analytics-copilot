"""
LangGraph Hybrid Agent for Retail Analytics
Orchestrates RAG, SQL, and synthesis with repair loops
"""
from typing import TypedDict, Any, List, Annotated
import operator
from langgraph.graph import StateGraph, END
import json
import re

# Import our modules
from agent.dspy_signatures import setup_dspy, Router, NLToSQL, Synthesizer
from agent.rag.retrieval import get_retriever
from agent.tools.sqlite_tool import get_db


# ===== STATE DEFINITION =====
class AgentState(TypedDict):
    """State passed between nodes in the graph"""
    # Input
    question: str
    format_hint: str
    
    # Routing
    route: str
    
    # RAG
    doc_chunks: List[dict]
    
    # Planning
    planner_output: dict
    
    # SQL
    sql_query: str
    sql_result: dict
    
    # Output
    final_answer: Any
    confidence: float
    explanation: str
    citations: List[str]
    
    # Control
    errors: Annotated[List[str], operator.add]
    repair_count: int
    trace: Annotated[List[str], operator.add]


# ===== INITIALIZE COMPONENTS =====
print("Initializing agent components...")
setup_dspy()
router = Router()
nl_to_sql = NLToSQL()
synthesizer = Synthesizer()
retriever = get_retriever()
db = get_db()
schema = db.get_schema()
print("Agent components ready!")


# ===== NODE FUNCTIONS =====

def router_node(state: AgentState) -> AgentState:
    """Route the question to appropriate strategy"""
    print(f"\n[ROUTER] Classifying question...")
    
    route = router.forward(state["question"])
    state["route"] = route
    state["trace"].append(f"Router: {route}")
    
    print(f"[ROUTER] Route determined: {route}")
    return state


def retriever_node(state: AgentState) -> AgentState:
    """Retrieve relevant document chunks"""
    print(f"\n[RETRIEVER] Searching documents...")
    
    chunks = retriever.retrieve(state["question"], top_k=3)
    state["doc_chunks"] = chunks
    state["trace"].append(f"Retrieved {len(chunks)} chunks")
    
    print(f"[RETRIEVER] Found {len(chunks)} relevant chunks")
    for chunk in chunks:
        print(f"  - {chunk['id']} (score: {chunk['score']:.3f})")
    
    return state


def planner_node(state: AgentState) -> AgentState:
    """Extract constraints from documents (dates, categories, formulas)"""
    print(f"\n[PLANNER] Extracting constraints from documents...")
    
    # Extract key information from doc chunks
    doc_text = "\n".join([chunk["content"] for chunk in state["doc_chunks"]])
    
    constraints = {
        "dates": [],
        "categories": [],
        "formulas": [],
        "context": doc_text[:500]  # Keep first 500 chars
    }
    
    # Extract date ranges (YYYY-MM-DD format)
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', doc_text)
    constraints["dates"] = dates
    
    # Extract category mentions
    categories = ["Beverages", "Condiments", "Confections", "Dairy Products", 
                  "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"]
    for cat in categories:
        if cat in doc_text:
            constraints["categories"].append(cat)
    
    # Extract formulas (lines with = signs)
    for line in doc_text.split('\n'):
        if '=' in line and any(word in line for word in ['SUM', 'COUNT', 'AOV', 'GM']):
            constraints["formulas"].append(line.strip())
    
    state["planner_output"] = constraints
    state["trace"].append(f"Planner: extracted {len(dates)} dates, {len(constraints['categories'])} categories")
    
    print(f"[PLANNER] Extracted constraints:")
    print(f"  Dates: {dates}")
    print(f"  Categories: {constraints['categories']}")
    print(f"  Formulas: {len(constraints['formulas'])}")
    
    return state


def nl_to_sql_node(state: AgentState) -> AgentState:
    """Generate SQL query from natural language"""
    print(f"\n[NL-TO-SQL] Generating SQL query...")
    
    # Build context from planner output
    context_parts = []
    if state.get("planner_output"):
        plan = state["planner_output"]
        if plan.get("dates"):
            context_parts.append(f"Date range: {plan['dates']}")
        if plan.get("categories"):
            context_parts.append(f"Categories: {', '.join(plan['categories'])}")
        if plan.get("context"):
            context_parts.append(f"Context: {plan['context'][:200]}")
    
    context = "\n".join(context_parts)
    
    # Generate SQL
    sql = nl_to_sql.forward(
        question=state["question"],
        schema=schema,
        context=context
    )
    
    state["sql_query"] = sql
    state["trace"].append(f"Generated SQL: {sql[:100]}...")
    
    print(f"[NL-TO-SQL] Generated query:")
    print(f"  {sql}")
    
    return state


def executor_node(state: AgentState) -> AgentState:
    """Execute SQL query"""
    print(f"\n[EXECUTOR] Running SQL query...")
    
    result = db.execute_query(state["sql_query"])
    
    # Add tables_used for citations
    if result["success"] and result["rows"]:
        # Extract table names from SQL
        sql_upper = state["sql_query"].upper()
        tables_used = []
        for table in ["ORDERS", "ORDER DETAILS", "PRODUCTS", "CUSTOMERS", "CATEGORIES"]:
            if table in sql_upper:
                # Convert to proper case
                if table == "ORDER DETAILS":
                    tables_used.append("Order Details")
                else:
                    tables_used.append(table.title())
        result["tables_used"] = tables_used
    
    state["sql_result"] = result
    state["trace"].append(f"SQL execution: {'success' if result['success'] else 'failed'}")
    
    if result["success"]:
        print(f"[EXECUTOR] Query succeeded, {len(result['rows'])} rows returned")
    else:
        print(f"[EXECUTOR] Query failed: {result['error']}")
        state["errors"].append(result["error"])
    
    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesize final answer from SQL results and documents"""
    print(f"\n[SYNTHESIZER] Creating final answer...")
    
    result = synthesizer.forward(
        question=state["question"],
        format_hint=state["format_hint"],
        sql_result=state.get("sql_result"),
        doc_chunks=state.get("doc_chunks", [])
    )
    
    state["final_answer"] = result["final_answer"]
    state["explanation"] = result["explanation"]
    state["confidence"] = result["confidence"]
    state["citations"] = result["citations"]
    state["trace"].append(f"Synthesized answer: {result['final_answer']}")
    
    print(f"[SYNTHESIZER] Final answer: {result['final_answer']}")
    print(f"[SYNTHESIZER] Confidence: {result['confidence']:.2f}")
    
    return state


def repair_node(state: AgentState) -> AgentState:
    """Attempt to repair failed SQL or invalid output"""
    print(f"\n[REPAIR] Attempting repair (attempt {state['repair_count'] + 1}/2)...")
    
    state["repair_count"] += 1
    state["trace"].append(f"Repair attempt {state['repair_count']}")
    
    # If SQL failed, try to fix it
    if state.get("sql_result") and not state["sql_result"]["success"]:
        error = state["sql_result"]["error"]
        print(f"[REPAIR] Fixing SQL error: {error}")
        
        # Simple repair strategies
        sql = state["sql_query"]
        
        # Strategy 1: Fix table name case issues
        if "no such table" in error.lower():
            # Try wrapping table names in quotes
            for table in ["Order Details", "Orders", "Products", "Customers"]:
                sql = sql.replace(table, f'"{table}"')
        
        # Strategy 2: Fix column name issues
        if "no such column" in error.lower():
            # Extract mentioned column and try common alternatives
            pass  # Could implement smart column name mapping
        
        state["sql_query"] = sql
        state["trace"].append(f"Repaired SQL: {sql[:100]}...")
    
    return state


def validation_node(state: AgentState) -> AgentState:
    """Validate output format matches format_hint"""
    print(f"\n[VALIDATOR] Checking output format...")
    
    format_hint = state["format_hint"]
    answer = state["final_answer"]
    
    valid = True
    
    # Type checking
    if format_hint == "int":
        valid = isinstance(answer, int)
    elif format_hint == "float":
        valid = isinstance(answer, (int, float))
    elif "{" in format_hint:
        valid = isinstance(answer, dict)
    elif "list" in format_hint:
        valid = isinstance(answer, list)
    
    if not valid:
        print(f"[VALIDATOR] Format mismatch! Expected {format_hint}, got {type(answer)}")
        state["errors"].append(f"Format mismatch: expected {format_hint}, got {type(answer)}")
    else:
        print(f"[VALIDATOR] Format validated: {format_hint}")
    
    state["trace"].append(f"Validation: {'passed' if valid else 'failed'}")
    return state


# ===== CONDITIONAL EDGES =====

def should_retrieve(state: AgentState) -> str:
    """Decide if we need to retrieve documents"""
    route = state["route"]
    if route in ["rag", "hybrid"]:
        return "retrieve"
    else:
        return "plan"


def should_repair(state: AgentState) -> str:
    """Decide if we need to repair"""
    # Check if SQL failed
    if state.get("sql_result") and not state["sql_result"]["success"]:
        if state["repair_count"] < 2:
            return "repair"
    
    # Check if format is wrong
    format_hint = state["format_hint"]
    answer = state.get("final_answer")
    
    format_valid = True
    if format_hint == "int" and not isinstance(answer, int):
        format_valid = False
    elif format_hint == "float" and not isinstance(answer, (int, float)):
        format_valid = False
    elif "{" in format_hint and not isinstance(answer, dict):
        format_valid = False
    elif "list" in format_hint and not isinstance(answer, list):
        format_valid = False
    
    if not format_valid and state["repair_count"] < 2:
        return "repair"
    
    return "end"


def after_repair(state: AgentState) -> str:
    """After repair, decide next step"""
    if state["repair_count"] >= 2:
        # Max repairs reached, go to synthesis anyway
        return "synthesize"
    else:
        # Try executing SQL again
        return "execute"


# ===== BUILD GRAPH =====

def create_graph():
    """Build and compile the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route", router_node)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("plan", planner_node)
    workflow.add_node("nl_to_sql", nl_to_sql_node)
    workflow.add_node("execute", executor_node)
    workflow.add_node("synthesize", synthesizer_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("validate", validation_node)
    
    # Set entry point
    workflow.set_entry_point("route")
    
    # Add edges
    workflow.add_conditional_edges(
        "route",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "plan": "plan"
        }
    )
    
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", "nl_to_sql")
    workflow.add_edge("nl_to_sql", "execute")
    
    workflow.add_conditional_edges(
        "execute",
        lambda state: "synthesize" if state["sql_result"]["success"] else "repair",
        {
            "synthesize": "synthesize",
            "repair": "repair"
        }
    )
    
    workflow.add_conditional_edges(
        "repair",
        after_repair,
        {
            "execute": "execute",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("synthesize", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        should_repair,
        {
            "repair": "repair",
            "end": END
        }
    )
    
    return workflow.compile()


# ===== MAIN EXECUTION FUNCTION =====

def run_agent(question: str, format_hint: str) -> dict:
    """
    Run the agent on a single question
    
    Returns:
        dict with output contract fields
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING QUESTION: {question}")
    print(f"EXPECTED FORMAT: {format_hint}")
    print(f"{'='*60}")
    
    # Initialize state
    initial_state = {
        "question": question,
        "format_hint": format_hint,
        "route": "",
        "doc_chunks": [],
        "planner_output": {},
        "sql_query": "",
        "sql_result": {},
        "final_answer": None,
        "confidence": 0.0,
        "explanation": "",
        "citations": [],
        "errors": [],
        "repair_count": 0,
        "trace": []
    }
    
    # Create and run graph
    graph = create_graph()
    final_state = graph.invoke(initial_state)
    
    # Build output
    output = {
        "id": "",  # Will be set by caller
        "final_answer": final_state.get("final_answer"),
        "sql": final_state.get("sql_query", ""),
        "confidence": final_state.get("confidence", 0.0),
        "explanation": final_state.get("explanation", ""),
        "citations": final_state.get("citations", [])
    }
    
    print(f"\n{'='*60}")
    print(f"EXECUTION TRACE:")
    for step in final_state["trace"]:
        print(f"  - {step}")
    print(f"{'='*60}\n")
    
    return output


# ===== TESTING =====
if __name__ == "__main__":
    print("Testing Hybrid Agent Graph...\n")
    
    # Test question
    test_question = "According to the product policy, what is the return window (days) for unopened Beverages? Return an integer."
    test_format = "int"
    
    result = run_agent(test_question, test_format)
    
    print("\n=== FINAL OUTPUT ===")
    print(json.dumps(result, indent=2))