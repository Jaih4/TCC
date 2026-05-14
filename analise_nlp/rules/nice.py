def check_nice(text: str) -> bool:
    # elogios gerais 
    nice_keywords = ['ótimo', 'excelente', 'maravilhoso']
    return any(word in text.lower() for word in nice_keywords)