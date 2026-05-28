import re as regex

def check_spam(text: str) -> bool:
    """
    Retorna True se QUALQUER palavra for repetida 3 ou mais vezes consecutivas.
    Ignora diferenças entre maiúsculas e minúsculas (ex: "Oi oi OI" -> True).
    """

    pattern = r'\b(\w+)\b(?:\s+\1\b){3,}'
    

    if regex.search(pattern, text, regex.IGNORECASE):
        return True
        
    return False