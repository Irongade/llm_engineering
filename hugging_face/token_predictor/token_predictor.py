from typing import List, Dict
import math
from openai import OpenAI

class TokenPredictor:
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.messages = []
        self.predictions = []
        self.model_name = model_name

    def predict_tokens(self, prompt: str, max_tokens: int = 100) -> List[Dict]:
        """
        Generate text token by token and track prediction probabilities.
        Returns list of predictions with top token and alternatives.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,  # Use temperature 0 for deterministic output
            logprobs=True,
            seed=42,
            top_logprobs=3,  # Get top 3 token predictions
            stream=True,  # Stream the response
        )

        predictions = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                logprobs = chunk.choices[0].logprobs.content[0].top_logprobs
                logprob_dict = {item.token: item.logprob for item in logprobs}

                # Get top predicted token and probability
                top_token = token
                top_prob = logprob_dict[token]

                # Get alternative predictions
                alternatives = []
                for alt_token, alt_prob in logprob_dict.items():
                    if alt_token != token:
                        alternatives.append((alt_token, math.exp(alt_prob)))
                alternatives.sort(key=lambda x: x[1], reverse=True)

                prediction = {
                    "token": top_token,
                    "probability": math.exp(top_prob),
                    "alternatives": alternatives[:2],  # Keep top 2 alternatives
                }
                predictions.append(prediction)

        return predictions

