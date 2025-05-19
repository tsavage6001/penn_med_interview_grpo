import numpy as np
import pandas as pd
from typing import List, Dict
from openai import OpenAI
import os
import openai
import difflib 

class RewardCalculator:
    
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAIKEY"))
        self.medical_data = pd.DataFrame({
            'case': [
                "I've had persistent headaches and nausea for two weeks",
                "I experience chest pain when climbing stairs",
                "I'm unusually thirsty and urinating frequently",
                "I've been feeling extremely tired for months",
                "I have sharp abdominal pain after eating"
            ],
            'diagnosis': [
                "Elevated intracranial pressure",
                "Stable Angina",
                "diabetes mellitus",
                "hypothyroidism",
                "peptic ulcer"
            ]
        })

    def get_llm_diagnosis(self, conversation: str) -> str:
        """Get diagnosis from GPT-4 based on conversation"""
        prompt = f"""Based on the following doctor-patient conversation, 
        what is the most likely diagnosis? Provide only the diagnosis name.
        
        Conversation:
        {conversation}
        
        Diagnosis:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()

    import difflib

    def compare_diagnoses(self, llm_diagnosis: str, case: str) -> float:
        """Compare LLM diagnosis with expected diagnosis using fuzzy matching and LLM response."""
        # Fuzzy match the case
        cases = self.medical_data['case'].tolist()
        match = difflib.get_close_matches(case, cases, n=1)
        if match:
            expected = self.medical_data[self.medical_data['case'] == match[0]]['diagnosis'].iloc[0].lower()
        else:
            print(f"Warning: No close match found for case: {case!r}")
            return 0.0
    
        prompt = f"""Compare these two diagnoses. If they are approximately the same (ex heart attack and myocardial infarction) write a one word answer of "same", if they are different write a one word answer of "different": 
    Diagnosis 1: {llm_diagnosis}
    Diagnosis 2: {expected}
    """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip().lower()
        if "same" in result:
            return 1.0
        elif "different" in result:
            return 0.0
        else:
            print(f"Warning: Unexpected LLM response: {result!r}")
            return 0.0

    def calculate_reward(
        self,
        doctor_response: str,
        sibling_responses: List[str],
        full_conversation: str,
        initial_case: str
    ) -> float:
        """Calculate reward incorporating diagnosis accuracy"""
        # Get diagnosis scores for current response and siblings
        llm_diagnosis = self.get_llm_diagnosis(full_conversation)
        diagnosis_score = self.compare_diagnoses(llm_diagnosis, initial_case)
        
        # Get diagnosis scores for siblings
        sibling_diagnosis_scores = []
        for sibling in sibling_responses:
            sibling_diag = self.get_llm_diagnosis(sibling)
            sibling_score = self.compare_diagnoses(sibling_diag, initial_case)
            sibling_diagnosis_scores.append(sibling_score)
        
        # Calculate relative score based on diagnosis accuracy
        if not sibling_diagnosis_scores:
            relative_score = diagnosis_score
        else:
            mean_sibling_score = np.mean(sibling_diagnosis_scores)
            relative_score = 0.5 + 0.5 * (diagnosis_score - mean_sibling_score)
        
        return relative_score
