import json
import re
from datetime import datetime
from typing import List, Literal, Optional, Dict, Union

from pydantic import BaseModel, Field, ConfigDict, model_validator, ValidationInfo, ValidationError, AfterValidator
from typing_extensions import Annotated

from agent import Agent
from ragatouille import RAGPretrainedModel
from rerankers import Reranker, Document

# Load the RAG model
colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# ranker =  Reranker("flashrank")
class CodeChoice(BaseModel):
    code: str
    description: str

class DiagnosisRanking(BaseModel):
    top_one: CodeChoice = Field(..., description="Most suitable ICD code")

class ControlGroupOutput(BaseModel):
    top_one: str = Field(..., description="Most suitable ICD code, format example: A0103, B081, etc  ")


class Codify:
    def __init__(self):

        self.icd_database = RAGPretrainedModel.from_index('.ragatouille/colbert/indexes/icd10_index')

        self.simple_rerank_agent = Agent(response_model=DiagnosisRanking,
                                       ai_provider="groq_client",
                                       model="llama3.1")

        self.control_group_agent = Agent(response_model=ControlGroupOutput,
                                       ai_provider="groq_client",
                                       model="llama-3.1-8b-instant")




    def simple_rerank(self, query:str, icd_references: List[Dict]):
        system_prompt = """
        You are a medical coding expert that can rerank ICD-10 codes based on their relevance to a query and a list of ICD-10 references.
        You will be given a query and a list of ICD-10 references.
        You must rerank the ICD-10 references based on their relevance to the query.
        You must return only the top 1 ICD-10 reference.
        You must return the passage that lead to the most relevant ICD-10 references from the query.
        Do not include your resoning in your response, just return the ICD-10 code and the content of the ICD-10 reference.
        The response should be in the following format:
         ""{
            "code": "A0103",
            "content": "Acute myocardial infarction",
            "keywords": ["passage1", "passage2", "passage3"]
            }
        ""
        Do not return any other information.
        """
        context = f"""
        query: {query}
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
        # documents = [Document(text=doc["content"], metadata=doc["document_metadata"], doc_id=doc["document_id"]) for doc in icd_references[:2]]
        # docs = ranker.rank(query, docs=documents)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@', json.dumps(icd_references, indent=2))
        # result = docs.top_k(1)
        # Rank ICD codes
        ranked_codes = self.simple_rerank(query, icd_references)
        print('@@@@@@@@@@@@@@@@@@@@@@@', ranked_codes)
        result = json.loads(ranked_codes["choices"][0]["message"]["content"].strip('\n'))
        code = result["code"].strip()
        description = result["content"].strip()
        # return ranked_codes
        return {"top_one": {"code": code, "description": description}}


    def get_control_group_output(self, query: str):
        system_prompt = "You are a medical coding expert that can suggest a control group for a given query. ICD-10-CM code"
        response = self.simple_rerank_agent.inference(system_prompt, query)
        return response
