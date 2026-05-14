def check_caps(text: str) -> bool:
    # Verifica se mais de 50% do texto está em caixa alta (evita falsos positivos com siglas)
    upper_chars = sum(1 for c in text if c.isupper())
    return upper_chars > (len(text) / 2) and len(text) > 5