def check_middle(text: str) -> bool:
    # Palavrões de intensificação (puta, foda, caralho)
    middle_keywords = ['puta', 'foda', 'caralho', 'merda']
    return any(word in text.lower() for word in middle_keywords)