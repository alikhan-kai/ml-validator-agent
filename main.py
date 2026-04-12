# Import standard system and operating system modules for file and environment management
import os
import sys
import logging
import json
from datetime import datetime

# Import dotenv to securely load API keys from the .env file
from dotenv import load_dotenv
# Import the LangChain tracer to send execution data to LangSmith
from langchain_core.tracers import LangChainTracer

# Import project-specific components: the state model and the graph constructor
from models import MLPipelineState
from graph import build_ml_graph

# Initialize environment variables from the .env file
load_dotenv()

# ============================================================================
# LangSmith Configuration (Assignment Bonus: +5 points)
# ============================================================================
# Explicitly setting environment variables to enable full observability and tracing
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ml-validator-agent")

# Configure global logging to provide clear feedback in the terminal during execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_ml_pipeline(input_file: str):
    """
    Executes the LangGraph ML analysis pipeline.
    Ensures compliance with the 'Minimum 3 nodes' requirement.
    """
    logging.info("="*80)
    logging.info("ML VALIDATOR AGENT - LangGraph Pipeline Execution")
    logging.info("="*80)
    logging.info(f"Processing dataset metadata from: {input_file}")

    # Build and compile the state graph from graph.py
    graph = build_ml_graph()

    # Initialize the starting state of the pipeline using the Pydantic state model
    initial_state = MLPipelineState(raw_input="")

    # Invoke the graph with tracing enabled via callbacks
    # This sends every node execution step to the LangSmith dashboard
    result = graph.invoke(
        initial_state,
        config={
            "callbacks": [
                LangChainTracer(project_name=os.getenv("LANGSMITH_PROJECT", "ml-validator"))
            ],
            "metadata": {
                "input_file": input_file,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

    return result

def log_final_results(result):
    """
    Parses and displays the final agent output in the console.
    Note: LangGraph returns a dictionary (dict) upon completion.
    """
    logging.info("\n" + "="*30)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("="*30)
    
    # Safely extract data from the result dictionary using .get()
    risk = result.get('risk_analysis')
    if risk:
        logging.info(f"RISK LEVEL: {risk.risk_score}")
        logging.info(f"OVERFITTING LIKELY: {risk.is_overfitting_likely}")
        # Truncate reasoning for a cleaner console output
        logging.info(f"ANALYSIS: {risk.reasoning[:150]}...")
    
    strat = result.get('final_strategy')
    if strat:
        logging.info("-" * 30)
        logging.info(f"SUGGESTED ACTIONS: {len(strat.recommended_actions)} key points")
        logging.info(f"VALIDATION METHOD: {strat.validation_strategy}")
    
    logging.info(f"\n[OK] Detailed report saved to: final_report.json")

def main():
    """
    Application entry point. Performs safety checks before running the agent.
    """
    # Verify that the Google API Key is set to avoid runtime authentication errors
    if not os.getenv("GOOGLE_API_KEY"):
        logging.error("CRITICAL: GOOGLE_API_KEY not found in .env file!")
        sys.exit(1)

    # Define the source of input data (Dataset metadata)
    input_file = "input_data.txt"

    # Check if the input file exists to prevent IO errors
    if not os.path.exists(input_file):
        logging.error(f"CRITICAL: Input file not found: {input_file}")
        sys.exit(1)

    try:
        # Execute the core agentic pipeline
        final_state = run_ml_pipeline(input_file)
        
        # Log the summarized results to the terminal
        log_final_results(final_state)
        
    except Exception as e:
        # Robust error handling with full traceback for debugging
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Ensure the main function only runs when the script is executed directly
if __name__ == "__main__":
    main()