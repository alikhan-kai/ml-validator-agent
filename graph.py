# Import standard logging to track the execution flow in the console
import logging
# Import json for structured data serialization and saving reports
import json
# Import typing for explicit type hinting and better code readability
from typing import Dict, Any
# Import LangGraph core components for building the state-based workflow
from langgraph.graph import StateGraph, END

# Import custom State model and LLM interface functions from local modules
from models import MLPipelineState
from llm_functions import get_risk_analysis, get_feature_strategy

# ============================================================================
# Node 1: Ingest ML Project Data (Logic)
# This node is responsible for the initial data loading phase
# ============================================================================
def n_ingest_data(state: MLPipelineState) -> Dict[str, Any]:
    # Log the start of the data ingestion process
    logging.info("[NODE: ingest_data] Reading input_data.txt")
    
    # Attempt to read the dataset metadata from the local text file
    try:
        # Open file with utf-8 encoding to handle all types of characters
        with open("input_data.txt", "r", encoding="utf-8") as f:
            content = f.read()
    # Fallback logic if the file is missing: use existing state or a default string
    except FileNotFoundError:
        content = state.raw_input if state.raw_input else "No data provided"

    # Log completion of the loading phase with the character count for verification
    logging.info(f"[NODE: ingest_data] Successfully loaded {len(content)} characters")

    # Return a dictionary that updates the current state of the graph
    return {
        "raw_input": content, # Updates the main input text
        "progress": 30       # Sets progress bar to 30%
    }

# ============================================================================
# Node 2: Statistical Risk Analysis (LLM Call #1)
# This node executes the first AI call to identify potential ML risks
# ============================================================================
def n_analyze_risks(state: MLPipelineState) -> Dict[str, Any]:
    # Log the start of the risk assessment phase
    logging.info("[NODE: analyze_risks] Starting LLM risk assessment")
    
    # Invoke the LLM function from llm_functions.py passing the raw input text
    # This call returns a structured Pydantic model (ModelRiskAnalysis)
    risk_report = get_risk_analysis(state.raw_input)
    
    # Update the state with the Pydantic object and advance progress to 60%
    return {
        "risk_analysis": risk_report,
        "progress": 60
    }

# ============================================================================
# Node 3: Feature Strategy & Final Output (LLM Call #2)
# This node generates the action plan based on previously identified risks
# ============================================================================
def n_generate_strategy(state: MLPipelineState) -> Dict[str, Any]:
    # Log the start of the final planning phase
    logging.info("[NODE: generate_strategy] Starting LLM strategy planning")
    
    # Combine the original input and the risk analysis reasoning for the next LLM
    # This enables "Agentic" behavior by passing context between nodes
    risk_summary = f"Risk Score: {state.risk_analysis.risk_score}. Reasoning: {state.risk_analysis.reasoning}"
    
    # Execute the second LLM call to generate a tailored Feature Engineering plan
    strategy = get_feature_strategy(state.raw_input, risk_summary)
    
    # Construct the final JSON output structure
    # metadata can be adjusted based on the project type (e.g., Churn or Credit)
    final_output = {
        "metadata": "ML Model Validation Report",
        "risk": state.risk_analysis.model_dump(),  # Convert Pydantic object to dict
        "strategy": strategy.model_dump()          # Convert Pydantic object to dict
    }
    
    # Write the results to a physical file (acts as the primary deliverable)
    with open("final_report.json", "w", encoding="utf-8") as f:
        # Use indent=4 for a human-readable, pretty-printed JSON file
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    # Log successful report generation
    logging.info("[NODE: generate_strategy] Final report saved to final_report.json")

    # Complete the pipeline: progress reaches 100%
    return {
        "final_strategy": strategy,
        "progress": 100
    }

# ============================================================================
# Graph Builder
# Orchestrates the execution sequence: Ingest -> Analyze -> Plan
# ============================================================================
def build_ml_graph():
    # Initialize the state-driven graph using the custom State class
    workflow = StateGraph(MLPipelineState)

    # Register each functional node into the graph architecture
    workflow.add_node("ingest_data", n_ingest_data)
    workflow.add_node("analyze_risks", n_analyze_risks)
    workflow.add_node("generate_strategy", n_generate_strategy)

    # Define the sequential execution edges (The "Roadmap")
    # 1. Define where the agent starts
    workflow.set_entry_point("ingest_data")
    # 2. Define the path from data loading to analysis
    workflow.add_edge("ingest_data", "analyze_risks")
    # 3. Define the path from analysis to strategy generation
    workflow.add_edge("analyze_risks", "generate_strategy")
    # 4. Define the end of the process
    workflow.add_edge("generate_strategy", END)

    # Log that the structure is validated and compiled
    logging.info("[GRAPH] ML Validator Graph built successfully")
    
    # Compile the workflow into an executable LangGraph object
    return workflow.compile()