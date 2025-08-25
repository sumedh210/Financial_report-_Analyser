import os
import traceback
from groq import Groq
from config import MODEL_KEY

class HydeGenerator:
    def __init__(self, model: str="llama3-8b-8192"):
        if not MODEL_KEY:
            raise ValueError("MODEL_KEY is not set in environment variables.")

        self.client = Groq(api_key=MODEL_KEY)
        self.model = model

    def generate_hypothetical_answer(self, query:str)-> str:
        prompt =(
            "Please write a passage that answers the following question. "
            "The passage should be written in a formal tone, as if it were taken directly from a financial report or an analyst's summary.\n\n"
            f"Question: {query}\n\n"
            "Passage:"
        )
        

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"[HyDEGenerator Error] {e}")
            return "Error generating response"
        