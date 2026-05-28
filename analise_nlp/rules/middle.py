from .remover_acento import remover_acentos
import re as regex
def check_middle(text: str) -> bool:
    # Palavrões de intensificação (puta, foda, caralho)
    text = remover_acentos(text.lower())
    middle_keywords = [
        "bagulho",
        "caceta",
        "cacete",
        "caralhada",
        "caralho",
        "caralhos",
        "desgraca",
        "do cacete",
        "do cao",
        "do caralho",
        "do inferno",
        "filho da puta",
        "filhodaputa",
        "foda",
        "foda-se",
        "fodase",
        "fodastico",
        "fodido",
        "krl",
        "merda",
        "pica",
        "porra",
        "pqp",
        "pra caralho",
        "puta merda",
        "puta que o pariu",
        "puta que pariu",
        "puta",
        "putaria",
        "puto",
        "que merda",
        "que porra e essa",
        "que porra",
        "ta foda",
        "ta porra",
        "vai se foder",
        "vai tomar no cu",
        "vtnc"
    ]
        # Verifica cada palavra da lista usando Regex para casar apenas a palavra inteira
    for word in middle_keywords:
        # \b cria uma "fronteira", garantindo que "cu" só dê match se estiver isolado
        padrao = rf'\b{word}\b'
        if regex.search(padrao, text):
            return True
            
    return False