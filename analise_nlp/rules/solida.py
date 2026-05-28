from .remover_acento import remover_acentos
import re as regex
def check_solida(text: str) -> bool:
    # apoio geral (ótimo, excelente, maravilhoso)
    text = remover_acentos(text.lower())


    solida_keywords = [
        "acolhimento",
        "admiravel",
        "arrasou",
        "bravo",
        "brilhante",
        "compreensao",
        "conquista",
        "conte comigo",
        "coragem",
        "empatia",
        "espetacular",
        "estamos ao seu lado",
        "estamos com voce",
        "estamos juntos",
        "estou com voce",
        "excelente trabalho",
        "exemplo de pessoa",
        "felicidades",
        "forca",
        "forcas",
        "guerreira",
        "guerreiro",
        "humanidade",
        "inspiracao",
        "inspirador",
        "justica",
        "magnifico",
        "mandou bem",
        "merece muito"
        "merecido",
        "meu apoio",
        "meus sentimentos",
        "muita luz",
        "nao desista",
        "orgulho de voce",
        "orgulho",
        "parabens",
        "pra cima",
        "que noticia boa",
        "resiliencia",
        "resistencia",
        "respeito",
        "sinto muito"
        "solidaria",
        "solidario",
        "sucesso",
        "tamo junto",
        "tmj",
        "total apoio",
        "vai dar tudo certo",
        "verdade aparecendo",
        "vitoria",
        "viva"
    ]
        # Verifica cada palavra da lista usando Regex para casar apenas a palavra inteira
    for word in solida_keywords:
        # \b cria uma "fronteira", garantindo que "cu" só dê match se estiver isolado
        padrao = rf'\b{word}\b'
        if regex.search(padrao, text):
            return True
            
    return False