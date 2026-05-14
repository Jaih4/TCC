def check_hate(text: str) -> bool:
    # Palavras de ataque a minorias (racismo, homofobia, etc.)
    hate_keywords = ['lista', 'de', 'termos', 'aqui']
    return any(word in text.lower() for word in hate_keywords)