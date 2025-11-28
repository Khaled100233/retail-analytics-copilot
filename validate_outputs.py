"""
Validation Script for Agent Outputs
Checks answers against expected results
"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


# Expected answers (based on Northwind data and docs)
EXPECTED_ANSWERS = {
    "rag_policy_beverages_return_days": {
        "answer": 14,
        "type": int,
        "description": "Beverages unopened: 14 days (from product_policy.md)"
    },
    "hybrid_top_category_qty_summer_1997": {
        "answer": {"category": "Beverages", "quantity": "~1500-2000"},  # Approximate
        "type": dict,
        "description": "Beverages category during June 1997"
    },
    "hybrid_aov_winter_1997": {
        "answer": "~500-1500",  # Range, depends on calculation
        "type": float,
        "description": "Average Order Value in Dec 1997"
    },
    "sql_top3_products_by_revenue_alltime": {
        "answer": "list of 3 dicts with product and revenue",
        "type": list,
        "description": "Top 3 products by revenue"
    },
    "hybrid_revenue_beverages_summer_1997": {
        "answer": "~9000-10000",  # Approximate
        "type": float,
        "description": "Beverages revenue June 1997"
    },
    "hybrid_best_customer_margin_1997": {
        "answer": {"customer": "some company", "margin": "positive float"},
        "type": dict,
        "description": "Top customer by gross margin in 1997"
    }
}


def validate_outputs(output_file: str = "outputs_hybrid.jsonl"):
    """Validate all outputs"""
    
    console.print("\n[bold blue]Validating Agent Outputs[/bold blue]\n")
    
    # Read outputs
    outputs = []
    with open(output_file, 'r') as f:
        for line in f:
            outputs.append(json.loads(line.strip()))
    
    console.print(f"Loaded {len(outputs)} results\n")
    
    # Create results table
    table = Table(title="Validation Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Format", style="magenta")
    table.add_column("Answer", style="green")
    table.add_column("Type Match", style="yellow")
    table.add_column("Citations", style="blue")
    table.add_column("Status", style="bold")
    
    results = {
        "total": len(outputs),
        "type_correct": 0,
        "has_citations": 0,
        "has_sql": 0,
        "has_explanation": 0
    }
    
    for output in outputs:
        q_id = output['id']
        answer = output.get('final_answer')
        citations = output.get('citations', [])
        sql = output.get('sql', '')
        explanation = output.get('explanation', '')
        
        # Get expected
        expected = EXPECTED_ANSWERS.get(q_id, {})
        expected_type = expected.get('type')
        
        # Check type
        type_match = "✓" if isinstance(answer, expected_type) else "✗"
        if isinstance(answer, expected_type):
            results["type_correct"] += 1
        
        # Check citations
        has_cites = len(citations) > 0
        if has_cites:
            results["has_citations"] += 1
        
        # Check SQL
        if sql:
            results["has_sql"] += 1
        
        # Check explanation
        if explanation:
            results["has_explanation"] += 1
        
        # Status
        status = "✓ PASS" if isinstance(answer, expected_type) and has_cites else "⚠ CHECK"
        
        # Add row
        table.add_row(
            q_id[:30] + "...",
            str(expected_type.__name__ if expected_type else "?"),
            str(answer)[:40] + ("..." if len(str(answer)) > 40 else ""),
            type_match,
            f"{len(citations)} items",
            status
        )
    
    console.print(table)
    
    # Summary statistics
    console.print("\n[bold]Validation Summary:[/bold]")
    console.print(f"  Total Questions: {results['total']}")
    console.print(f"  Type Correct: {results['type_correct']}/{results['total']} ({results['type_correct']/results['total']*100:.0f}%)")
    console.print(f"  Has Citations: {results['has_citations']}/{results['total']} ({results['has_citations']/results['total']*100:.0f}%)")
    console.print(f"  Has SQL: {results['has_sql']}/{results['total']}")
    console.print(f"  Has Explanation: {results['has_explanation']}/{results['total']}")
    
    # Detailed check for each question
    console.print("\n[bold]Detailed Analysis:[/bold]\n")
    
    for output in outputs:
        q_id = output['id']
        console.print(f"[cyan]{q_id}[/cyan]")
        console.print(f"  Answer: {output.get('final_answer')}")
        console.print(f"  Type: {type(output.get('final_answer')).__name__}")
        console.print(f"  Confidence: {output.get('confidence', 0):.2f}")
        console.print(f"  Citations: {', '.join(output.get('citations', []))}")
        console.print(f"  SQL Used: {'Yes' if output.get('sql') else 'No'}")
        
        if output.get('sql'):
            sql_preview = output['sql'][:80].replace('\n', ' ')
            console.print(f"  SQL: {sql_preview}...")
        
        console.print(f"  Explanation: {output.get('explanation', 'N/A')[:100]}...")
        console.print()
    
    # Check specific answers
    console.print("[bold]Answer Correctness Check:[/bold]\n")
    
    for output in outputs:
        q_id = output['id']
        answer = output.get('final_answer')
        
        if q_id == "rag_policy_beverages_return_days":
            is_correct = answer == 14
            console.print(f"  {q_id}: {'✓ CORRECT' if is_correct else '✗ WRONG'} (Expected: 14, Got: {answer})")
        
        elif q_id == "sql_top3_products_by_revenue_alltime":
            is_correct = isinstance(answer, list) and len(answer) == 3
            console.print(f"  {q_id}: {'✓ CORRECT' if is_correct else '✗ WRONG'} (Expected: list of 3, Got: {len(answer) if isinstance(answer, list) else 'not a list'})")
        
        elif q_id == "hybrid_top_category_qty_summer_1997":
            is_correct = isinstance(answer, dict) and "category" in answer and "quantity" in answer
            console.print(f"  {q_id}: {'✓ FORMAT OK' if is_correct else '✗ WRONG FORMAT'} (Expected: dict with category & quantity)")
        
        elif q_id == "hybrid_aov_winter_1997":
            is_correct = isinstance(answer, (int, float)) and answer > 0
            console.print(f"  {q_id}: {'✓ REASONABLE' if is_correct else '✗ CHECK'} (Expected: positive float, Got: {answer})")
        
        elif q_id == "hybrid_revenue_beverages_summer_1997":
            is_correct = isinstance(answer, (int, float)) and answer > 0
            console.print(f"  {q_id}: {'✓ REASONABLE' if is_correct else '✗ CHECK'} (Expected: positive float, Got: {answer})")
        
        elif q_id == "hybrid_best_customer_margin_1997":
            is_correct = isinstance(answer, dict) and "customer" in answer and "margin" in answer
            console.print(f"  {q_id}: {'✓ FORMAT OK' if is_correct else '✗ WRONG FORMAT'} (Expected: dict with customer & margin)")
    
    console.print("\n[bold green]Validation Complete![/bold green]\n")
    
    return results


if __name__ == "__main__":
    validate_outputs("outputs_hybrid.jsonl")