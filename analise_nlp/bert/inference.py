from transformers import pipeline
import os

# Aponta para a pasta do seu modelo recém-treinado. 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = BASE_DIR / "analise_3_classes_peso"



classifier = pipeline("text-classification", model=MODEL_NAME)

def analyze_text(text: str) -> list:
    """Analisa texto usando o modelo treinado localmente."""
    return classifier(text)