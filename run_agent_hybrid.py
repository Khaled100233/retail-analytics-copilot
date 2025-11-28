"""
CLI Interface for Retail Analytics Hybrid Agent
Supports batch processing of questions from JSONL input
"""
import click
import json
from pathlib import Path
from agent.graph_hybrid import create_graph
from agent.dspy_signatures import setup_dspy


def validate_output_format(final_answer, format_hint: str) -> bool:
    """
    Validate that final_answer matches the expected format_hint type
    
    Args:
        final_answer: The answer to validate
        format_hint: Expected format (e.g., "int", "float", "{...}", "list[...]")
    
    Returns:
        True if format matches, False otherwise
    """
    try:
        if format_hint == "int":
            return isinstance(final_answer, int)
        elif format_hint == "float":
            return isinstance(final_answer, (int, float))
        elif format_hint.startswith("{") or "dict" in format_hint.lower():
            return isinstance(final_answer, dict)
        elif format_hint.startswith("list") or format_hint.startswith("["):
            return isinstance(final_answer, list)
        else:
            return isinstance(final_answer, str)
    except:
        return False


def process_question(graph, question_data: dict) -> dict:
    """
    Process a single question through the agent graph
    
    Args:
        graph: Compiled LangGraph agent
        question_data: Dict with 'id', 'question', 'format_hint'
    
    Returns:
        Dict with all required output fields
    """
    try:
        # Run the graph with complete initial state
        result = graph.invoke({
            "question": question_data["question"],
            "format_hint": question_data["format_hint"],
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
        })
        
        # Extract fields from final state
        output = {
            "id": question_data["id"],
            "final_answer": result.get("final_answer"),
            "sql": result.get("sql_query", ""),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
            "citations": result.get("citations", [])
        }
        
        # Validate format
        if not validate_output_format(output["final_answer"], question_data["format_hint"]):
            output["errors"] = [f"Output format mismatch: expected {question_data['format_hint']}, got {type(output['final_answer']).__name__}"]
        
        return output
    
    except Exception as e:
        # Return error response but continue batch processing
        return {
            "id": question_data["id"],
            "final_answer": None,
            "sql": "",
            "confidence": 0.0,
            "explanation": f"Error: {str(e)}",
            "citations": [],
            "errors": [str(e)]
        }


@click.command()
@click.option(
    '--batch',
    type=click.Path(exists=True),
    required=True,
    help='Path to input JSONL file with questions'
)
@click.option(
    '--out',
    type=click.Path(),
    required=True,
    help='Path to output JSONL file for results'
)
@click.option(
    '--model',
    default='phi3.5:3.8b-mini-instruct-q4_K_M',
    help='Ollama model to use'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Print detailed progress information'
)
def main(batch: str, out: str, model: str, verbose: bool):
    """
    Retail Analytics Hybrid Agent - Batch Processing CLI
    
    Processes questions from input JSONL and writes results to output JSONL.
    Each question is processed independently with full error handling.
    
    Example:
        python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
    """
    
    # Setup
    click.echo("=" * 70)
    click.echo("Retail Analytics Hybrid Agent - Batch Mode")
    click.echo("=" * 70)
    click.echo(f"Input:  {batch}")
    click.echo(f"Output: {out}")
    click.echo(f"Model:  {model}")
    click.echo("=" * 70)
    
    # Initialize DSPy and agent graph
    click.echo("\n[1/3] Initializing DSPy...")
    setup_dspy(model_name=model)
    
    click.echo("[2/3] Building agent graph...")
    graph = create_graph()
    
    # Load input questions
    click.echo(f"[3/3] Loading questions from {batch}...")
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    click.echo(f"\nLoaded {len(questions)} questions.\n")
    
    # Process each question
    results = []
    for i, question_data in enumerate(questions, 1):
        click.echo(f"Processing {i}/{len(questions)}: {question_data['id']}")
        
        if verbose:
            click.echo(f"  Q: {question_data['question'][:80]}...")
            click.echo(f"  Format: {question_data['format_hint']}")
        
        # Process question
        result = process_question(graph, question_data)
        results.append(result)
        
        # Show result summary
        if verbose:
            click.echo(f"  ✓ Answer: {result['final_answer']}")
            click.echo(f"  ✓ Confidence: {result['confidence']:.2f}")
            click.echo(f"  ✓ Citations: {len(result['citations'])}")
            if 'errors' in result:
                click.echo(f"  ⚠ Errors: {result['errors']}")
        else:
            status = "✓" if 'errors' not in result else "⚠"
            click.echo(f"  {status} Done (confidence: {result['confidence']:.2f})")
        
        click.echo()
    
    # Write output file
    click.echo(f"\nWriting results to {out}...")
    with open(out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Summary
    click.echo("\n" + "=" * 70)
    click.echo("BATCH PROCESSING COMPLETE")
    click.echo("=" * 70)
    
    success_count = sum(1 for r in results if 'errors' not in r)
    error_count = len(results) - success_count
    
    click.echo(f"Total questions: {len(results)}")
    click.echo(f"Successful:      {success_count}")
    click.echo(f"Errors:          {error_count}")
    
    if error_count > 0:
        click.echo(f"\nQuestions with errors:")
        for r in results:
            if 'errors' in r:
                click.echo(f"  - {r['id']}: {r['errors'][0][:60]}...")
    
    click.echo(f"\nResults saved to: {out}")
    click.echo("=" * 70)


if __name__ == "__main__":
    main()
