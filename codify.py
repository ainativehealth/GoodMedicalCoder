import json
import re
from datetime import datetime
from typing import List, Literal, Optional, Dict, Union

from pydantic import BaseModel, Field, ConfigDict, model_validator, ValidationInfo, ValidationError, AfterValidator
from typing_extensions import Annotated

from agent import Agent
from ragatouille import RAGPretrainedModel


class CodeChoice(BaseModel):
    code: str
    description: str

class DiagnosisRanking(BaseModel):
    top_one: CodeChoice = Field(..., description="Most suitable ICD code")
    
class Codify:
    def __init__(self):
        
        self.icd_database = RAGPretrainedModel.from_index('.ragatouille/colbert/indexes/icd10_index') 

        self.simple_rerank_agent = Agent(response_model=DiagnosisRanking,
                                       ai_provider="azure_client",
                                       model="gpt-3.5-turbo")
        
        self.control_group_agent = Agent(response_model=ControlGroupOutput,
                                       ai_provider="azure_client",
                                       model="gpt-3.5-turbo")

      

    
    def simple_rerank(self, description:str, icd_references: List[Dict]):
        system_prompt = """
        You are a medical coding expert that can rerank ICD-10 codes based on a description and a list of ICD-10 references.
        """
        context = f"""
        Description: {description}
        ICD-10 references: {json.dumps(icd_references, indent=2)}
        """
        response = self.simple_rerank_agent.inference(system_prompt, context)
        return response
    
    
    def get_icd_code(self, query: str) -> dict:
        results = self.icd_database.search(query, k = 15)
        if not results:
            raise ValueError(f"No ICD code found for query: {query}")
        return results
    

    def normalize_icd_code(self, code: str) -> str:
        return code.replace('.', '')
    

    def get_ranked_icd_codes(self, query: str):
        # Get ICD code references
        icd_references = self.get_icd_code(query)
        
        # Rank ICD codes
        ranked_codes = self.simple_rerank(query, icd_references)

        return ranked_codes
    

    def get_control_group_output(self, query: str):
        system_prompt = "You are a medical coding expert that can suggest a control group for a given query. ICD-10-CM code"
        response = self.simple_rerank_agent.inference(system_prompt, query)
        return response
    
