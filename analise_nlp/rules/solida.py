def check_solida(text: str) -> bool:
    # apoio geral (ótimo, excelente, maravilhoso)
    solida_keywords = ['ótimo', 'excelente', 'maravilhoso']
    return any(word in text.lower() for word in solida_keywords)