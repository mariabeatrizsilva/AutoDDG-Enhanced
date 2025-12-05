import re
import json
from typing import Dict, List
import numpy as np
from collections import Counter


##---- Coverage Score 

'''
Run an LLM that extracts info about the dataset 
This is data to measure coverage in 5 coverage categories: 
- basic_info (15%)
- data_characteristics (25%)
- provenance (20%)
- usage_context (20%)
- quality_and_limitations (20%)

'''

class CoverageScorer:
    def __init__(self):
        
        self.coverage_dimensions = {
            'basic_info': {
                'weight': 0.20,
                'fields': [
                    'domain_or_field',
                    'primary_purpose'
                ]
            },
            'data_characteristics': {
                'weight': 0.30,
                'fields': [
                    'size_or_scale',
                    'data_format',
                    'data_types',
                    'temporal_coverage',
                    'sample_unit' 
                ]
            },
            'provenance': {
                'weight': 0.20,
                'fields': [
                    'collection_method',
                    'data_source',
                    'collection_date',
                    'preprocessing_steps'
                ]
            },
            'usage_context': {
                'weight': 0.20,
                'fields': [
                    'typical_applications',
                    'research_questions_addressed',
                    'benchmark_or_evaluation_role'
                ]
            },
            'quality_and_limitations': {
                'weight': 0.10,
                'fields': [
                    'known_limitations',
                    'biases_or_caveats',
                    'quality_issues',
                    'challenges_in_use'
                ]
            }
        }
    
    
    # CALCULATE COVERAGE SCORE : 
    def calculate_coverage(self, llm_result):
        """
        Score the LLM extraction result.
        
        Args:
            llm_result: Dict with dimensions as keys, each containing fields
            
        Returns:
            Dict with scores and details
        """
        dimension_scores = {}
        dimension_details = {}
        
        # Score each dimension
        for dimension, config in self.coverage_dimensions.items():
            score, details = self._score_dimension(
                llm_result.get(dimension, {}),
                config['fields']
            )
            dimension_scores[dimension] = score
            dimension_details[dimension] = details
        
        # Calculate weighted overall score
        overall_score = sum(
            dimension_scores[dim] * self.coverage_dimensions[dim]['weight']
            for dim in dimension_scores
        )
        
        metrics= {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'dimension_details': dimension_details
        }
        
        return metrics
    
    def _score_dimension(self, dimension_data, fields):
        """
        Score a single dimension by counting filled fields.
        
        Returns:
            tuple: (score, details_dict)
                - score: 0.0 to 1.0 (points / total_fields)
                - details: dict showing each field's status
        """
        points = 0
        details = {}
        
        for field in fields:
            value = dimension_data.get(field)
            is_filled = self._is_filled(value)
            
            if is_filled:
                points += 1
            
            details[field] = {
                'filled': is_filled,
                'value': value
            }
        
        score = points / len(fields) if fields else 0.0
        
        return score, details
    
    def _is_filled(self, value):
        """
        Check if a field is filled (returns 1) or empty (returns 0).
        """
        if value is None:
            return False
        
        if isinstance(value, str):
            value_clean = value.strip().lower()
            
            # Empty or placeholder phrases
            placeholders = [
                'not mentioned',
                'not specified',
                'not provided',
                'not available',
                'unknown',
                'n/a',
                'null',
                'none',
                ''
            ]
            
            if value_clean in placeholders:
                return False
            
            return True
        
        elif isinstance(value, (list, dict)):
            return len(value) > 0
        
        return True
    
    
    
    
    
    