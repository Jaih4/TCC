# analise_nlp/ensemble/scorer.py

from analise_nlp.rules.bad import check_bad
from analise_nlp.rules.hate import check_hate
from analise_nlp.rules.middle import check_middle
from analise_nlp.rules.nice import check_nice
from analise_nlp.rules.solida import check_solida
from analise_nlp.rules.caps import check_caps

def ensemble_score(text: str, all_texts_in_scrape: list, bert_scores: list) -> float:
    # --- PREPARAÇÃO DAS REGRAS ---
    rules = {
        'hate_list': check_hate(text),
        'bad_list': check_bad(text),
        'nice_list': check_nice(text),   # Suas funções nice/solida
        'solida_list': check_solida(text),
        'caps_lock': check_caps(text),
        'middlelist': check_middle(text),
        'spam': all_texts_in_scrape.count(text) > 3, # Exemplo simples de detecção de spam (mesmo texto repetido)
    }

    # --- INÍCIO DO FLUXOGRAMA ---
    p = 0
    s = 'pos' # Default conforme o "Neutro" do seu mapa

    # Classificador de Teor
    if rules['hate_list']:
        p, s = 4, 'neg'
    elif rules['bad_list']:
        p, s = 2, 'neg'
    elif rules['nice_list']:
        p, s = 2, 'pos'
    elif rules['solida_list']:
        p, s = 4, 'pos'

    # Classificação de Intensidade (Agregadores)
    agregadores = 0
    if rules['caps_lock']: agregadores += 1
    if rules['spam']: agregadores += 1
    if rules['middlelist']: agregadores += 1

    # Soma de intensidade (Regra: P = P + agregadores)
    p += agregadores # Simplifica a lógica do seu desenho (1, 2 ou 3)

    # Inversão inicial de sinal
    if s == 'neg':
        p = p * -1

    # --- CLASSIFICAÇÃO DE IRONIA (BERTimbau SOTA) ---
    # Aqui usamos o seu modelo treinado para checar se o sinal deve inverter
    bert_result = bert_scores[0]
    
    # Se BERT diz Tóxico (LABEL_1) mas o P é Positivo (Nice/Solida/Neutro)
    # OU se BERT diz Normal (LABEL_0) mas o P é Negativo (Hate/Bad)
    tem_ironia = False
    if (p > 0 and bert_result['label'] == 'LABEL_1') or \
       (p < 0 and bert_result['label'] == 'LABEL_0'):
        if bert_result['score'] > 0.75: # Confiança mínima para inverter
            tem_ironia = True

    if tem_ironia:
        # Lógica do diagrama: se neg faz módulo, se pos faz * -1
        p = p * -1 

    # --- RESULTADO FINAL (Escala 0 a 1) ---
    # No seu modelo, -7 é o pior caso (Hate + 3 agregadores)
    # +7 é o melhor caso (Solida + 3 agregadores)
    # Convertendo para 0.0 (Bom) a 1.0 (Ruim/Tóxico)
    normalized = 0.5 - (p / 14.0)
    
    return max(0.0, min(1.0, round(normalized, 4)))