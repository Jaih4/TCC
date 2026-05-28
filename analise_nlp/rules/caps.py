import re as regex

def check_caps(text: str) -> bool:
    # Extrai apenas as palavras (ignorando espaços e pontuações)
    words = regex.findall(r'\b\w+\b', text)
    return any(word.isupper() and len(word) >= 3 for word in words)