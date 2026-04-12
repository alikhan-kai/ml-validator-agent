from typing import Optional, List, Dict, Any
# Import Pydantic components for data validation and schema definition
from pydantic import BaseModel, Field, ConfigDict

# ============================================================================
# ML Analysis Models (Structured Output Schemas)
# These models define the exact JSON format the LLM must return.
# ============================================================================

class ModelRiskAnalysis(BaseModel):
    """
    Schema for the first LLM call: Statistical Overfitting Analysis.
    This ensures the AI provides a structured risk assessment.
    """
    # forbid extra fields to ensure the LLM doesn't hallucinate non-existent attributes
    model_config = ConfigDict(extra='forbid')

    risk_score: str = Field(
        description="The risk level detected: Low, Medium, or High"
    )
    is_overfitting_likely: bool = Field(
        description="Boolean flag indicating if the model configuration is prone to noise memorization"
    )
    reasoning: str = Field(
        description="Technical justification for the risk score, referencing feature-to-sample ratios or leakage"
    )


class FeatureStrategy(BaseModel):
    """
    Schema for the second LLM call: Technical Roadmap.
    Defines the specific engineering actions suggested by the AI.
    """
    model_config = ConfigDict(extra='forbid')

    recommended_actions: List[str] = Field(
        description="A prioritized list of feature engineering and cleaning steps"
    )
    suggested_features_count: int = Field(
        description="The target number of features after dimensionality reduction"
    )
    validation_strategy: str = Field(
        description="The optimal cross-validation approach (e.g., Stratified K-Fold, Time-Series Split)"
    )


# ============================================================================
# State for LangGraph
# This class acts as the 'shared memory' of your agent.
# It is passed between every node in the graph.
# ============================================================================

class MLPipelineState(BaseModel):
    """
    The central State object for the LangGraph workflow.
    It tracks the input, intermediate metadata, and final AI results.
    """
    
    # Stores the raw text loaded from input_data.txt
    raw_input: str = "" 
    
    # Stores parsed metadata like row counts and feature counts for quick access
    dataset_metadata: Dict[str, Any] = {}
    
    # Slots for the structured outputs from our two LLM nodes
    # They start as None and are filled as the graph progresses
    risk_analysis: Optional[ModelRiskAnalysis] = None
    final_strategy: Optional[FeatureStrategy] = None
    
    # A simple integer to track pipeline completion (0 to 100)
    progress: int = 0
    
    class Config:
        # Allows Pydantic to handle complex types if necessary
        arbitrary_types_allowed = True