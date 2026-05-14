import re

def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text.strip()