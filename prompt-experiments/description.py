import json
import csv
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ProfileStats:
    """Holds specific fields extracted from the Related_Profile_JSON."""
    summary: str
    source_length: int
    full_source_length: int

@dataclass
class DescriptionEntry:
    # --- Identifiers ---
    test_id: str
    dataset_name: str
    dataset_id: str
    
    # --- Prompt Info ---
    prompt: str  # Now just a simple string
    
    # --- Metadata ---
    description_source: str
    extraction_strategy: str
    
    # --- Profile Stats (Parsed from JSON) ---
    profile_stats: ProfileStats
    
    # --- Quality Scores ---
    completeness_score: float
    conciseness_score: float
    readability_score: float
    
    # --- Metrics (Stored as Dictionaries/Hash Maps) ---
    bert_metrics: Dict[str, float] = field(default_factory=dict)
    rouge_metrics: Dict[str, float] = field(default_factory=dict)
    strict_coverage: Dict[str, float] = field(default_factory=dict)
    lenient_coverage: Dict[str, float] = field(default_factory=dict)


def row_to_object(row):
    # 1. Parse the JSON string from the "Related_Profile_JSON" column
    try:
        raw_json = json.loads(row['Related_Profile_JSON'])
        
        # Extract only the fields you asked for
        profile_obj = ProfileStats(
            summary=raw_json.get('summary', ""),
            source_length=int(raw_json.get('source_length', 0)),
            full_source_length=int(raw_json.get('full_source_length', 0))
        )
    except (json.JSONDecodeError, TypeError):
        # Fallback if JSON is broken or empty
        profile_obj = ProfileStats(summary="", source_length=0, full_source_length=0)

    # 2. Build the main object
    return DescriptionEntry(
        test_id=row['Test_ID'],
        dataset_name=row['Dataset_Name'],
        dataset_id=row['dataset_id'],
        prompt=row['Prompt_Type'],  # Simple string
        description_source=row['Description_Source'],
        extraction_strategy=row['Extraction_Strategy'],
        
        # Add the parsed profile object
        profile_stats=profile_obj,
        
        completeness_score=float(row['Completeness_Score']),
        conciseness_score=float(row['Conciseness_Score']),
        readability_score=float(row['Readability_Score']),
        
        bert_metrics={
            'precision': float(row['bert_precision']),
            'recall':    float(row['bert_recall']),
            'f1':        float(row['bert_f1'])
        },
        
        rouge_metrics={
            'rouge1':    float(row['rouge1']),
            'rouge2':    float(row['rouge2']),
            'rougeL':    float(row['rougeL']),
            'rougeLsum': float(row['rougeLsum'])
        },
        
        strict_coverage={
            'overall':                 float(row['strict_coverage_overall']),
            'basic_info':              float(row['strict_coverage_basic_info']),
            'data_characteristics':    float(row['strict_coverage_data_characteristics']),
            'provenance':              float(row['strict_coverage_provenance']),
            'usage_context':           float(row['strict_coverage_usage_context']),
            'quality_and_limitations': float(row['strict_coverage_quality_and_limitations'])
        },
        
        lenient_coverage={
            'overall':                 float(row['lenient_coverage_overall']),
            'basic_info':              float(row['lenient_coverage_basic_info']),
            'data_characteristics':    float(row['lenient_coverage_data_characteristics']),
            'provenance':              float(row['lenient_coverage_provenance']),
            'usage_context':           float(row['lenient_coverage_usage_context']),
            'quality_and_limitations': float(row['lenient_coverage_quality_and_limitations'])
        }
    )