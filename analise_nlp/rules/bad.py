def check_bad(text: str) -> bool:
    # Ofensas gerais (lixo, imbecil, idiota)
    bad_keywords = ['lixo', 'imbecil', 'burro']
    return any(word in text.lower() for word in bad_keywords)