# Import os to access environment variables from the system
import os
# Import logging to provide real-time updates in the terminal during LLM calls
import logging
# Import the specialized LangChain class for interacting with Google's Gemini models
from langchain_google_genai import ChatGoogleGenerativeAI
# Import the traceable decorator to enable deep monitoring in LangSmith
from langsmith import traceable
# Import load_dotenv to read the .env file and load API keys into the environment
from dotenv import load_dotenv

# Import pre-defined prompts and Pydantic models for structured data validation
from prompts import ANALYZE_ML_RISKS, GENERATE_FEATURE_STRATEGY
from models import ModelRiskAnalysis, FeatureStrategy

# Execute load_dotenv to ensure all keys are available for the application
load_dotenv()

# ============================================================================
# Gemini LLM Initialization
# ============================================================================
# Initialize the Gemini model. We use "gemini-flash-latest" as it's optimized 
# for speed and cost while supporting advanced features like Structured Outputs.
# Temperature is set to 0 for maximum consistency and deterministic results.
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# ============================================================================
# Core Function for Structured LLM Responses
# ============================================================================
# This decorator registers this specific function call as a distinct step in LangSmith
@traceable(name="gemini_structured_call")
def call_gemini_structured(prompt: str, response_model):
    """
    Invokes Gemini and forces the response to adhere to a specific Pydantic schema.
    This fulfills the requirement for 'Structured Outputs' in the assignment.
    """
    # .with_structured_output() creates a wrapper that automatically 
    # parses the raw LLM string into a Pydantic object of the provided class.
    structured_llm = llm.with_structured_output(response_model)
    
    # Send the prompt to the model and return the validated Python object
    return structured_llm.invoke(prompt)

# ============================================================================
# Node Logic 2: Risk Analysis Function
# ============================================================================
@traceable(name="get_risk_analysis")
def get_risk_analysis(input_text: str) -> ModelRiskAnalysis:
    """
    First LLM Call: Evaluates dataset metadata for statistical risks like overfitting.
    """
    # Notify the user that the first AI reasoning phase has started
    logging.info("[LLM] Starting Risk Analysis via Gemini...")
    
    # Inject the input metadata into the pre-defined template from prompts.py
    formatted_prompt = ANALYZE_ML_RISKS.format(text=input_text)
    
    # Call the structured LLM function using the ModelRiskAnalysis Pydantic schema
    result = call_gemini_structured(formatted_prompt, ModelRiskAnalysis)
    
    # Log the summary of the AI's findings (e.g., Risk Score: High/Low)
    logging.info(f"[LLM] Risk Score detected: {result.risk_score}")
    return result

# ============================================================================
# Node Logic 3: Feature Strategy Generation
# ============================================================================
@traceable(name="get_feature_strategy")
def get_feature_strategy(input_text: str, risk_info: str) -> FeatureStrategy:
    """
    Second LLM Call: Develops a technical roadmap based on identified risks.
    """
    # Notify the user that the strategy planning phase has started
    logging.info("[LLM] Generating Feature Strategy via Gemini...")
    
    # Pass both the original input AND the previous risk analysis reasoning.
    # This chain of thought allows the agent to make data-driven decisions.
    formatted_prompt = GENERATE_FEATURE_STRATEGY.format(
        text=input_text, 
        risk_analysis=risk_info
    )
    
    # Call the LLM and parse the result into the FeatureStrategy Pydantic model
    result = call_gemini_structured(formatted_prompt, FeatureStrategy)
    
    # Log the final complexity of the suggested strategy
    logging.info(f"[LLM] Suggested features count: {result.suggested_features_count}")
    return result