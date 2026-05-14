from transformers import pipeline
import os

# Aponta para a pasta do seu modelo recém-treinado. 
MODEL_NAME = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'bert_sota_toxicidade_v2')

# MODEL_NAME = "C:/Users/danie/Documents/Python/analise_comentarios/bert_sota_toxicidade_v2"

classifier = pipeline("text-classification", model=MODEL_NAME)

def analyze_text(text: str) -> list:
    """Analisa texto usando o modelo treinado localmente."""
    return classifier(text)