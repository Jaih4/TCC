import unicodedata

def remover_acentos(texto):
    """Remove acentos de uma string."""
    if not isinstance(texto, str):
        return ""
    # Normaliza a string e remove os caracteres de acentuação
    return ''.join(c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c))