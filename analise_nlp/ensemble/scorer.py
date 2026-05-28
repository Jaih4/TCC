# analise_nlp/ensemble/scorer.py
from analise_nlp.rules.bad import check_bad
from analise_nlp.rules.hate import check_hate
from analise_nlp.rules.middle import check_middle
from analise_nlp.rules.nice import check_nice
from analise_nlp.rules.solida import check_solida
from analise_nlp.rules.caps import check_caps
from analise_nlp.rules.spam import check_spam

def ensemble_score(text: str, all_texts_in_scrape: list, bert_scores: list) -> float:
    # --- PREPARAÇÃO DAS REGRAS ---
    rules = {
        'hate_list': check_hate(text),
        'bad_list': check_bad(text),
        'nice_list': check_nice(text), 
        'solida_list': check_solida(text),
        'caps_lock': check_caps(text),
        'middlelist': check_middle(text),
        'spam': check_spam(text),
    }

    # Saldo começa zerado
    saldo = 0
    
    # Cada lista puxa o saldo para o seu lado
    if rules['hate_list']: saldo -= 4
    if rules['bad_list']:  saldo -= 2
    if rules['nice_list']: saldo += 2
    if rules['solida_list']: saldo += 4
    
    # Classificação de Intensidade (Agregadores)
    agregadores = 0
    if rules['caps_lock']: agregadores += 1
    if rules['spam']: agregadores += 1
    if rules['middlelist']: agregadores += 1

    # Define a Direção (s) e a Intensidade (p) separadamente
    if saldo > 0:
        s = 'pos'
        # Pega a base positiva e soma a força dos agregadores
        p = saldo + agregadores
        
    elif saldo < 0:
        s = 'neg'
        # Pega a base negativa (transformada em positivo pelo abs) e soma os agregadores
        p = abs(saldo) + agregadores
        
    else:
        # Se o saldo for 0 (neutro ou empate), os agregadores são IGNORADOS.
        # Afinal, gritar em CAPS LOCK uma frase neutra não a torna positiva nem negativa.
        p = 0
        s = 'neu'

    # Inversão inicial de sinal
    if s == 'neg':
        p = p * -1

    # --- CLASSIFICAÇÃO DE IRONIA (BERTimbau SOTA) ---
    # Aqui usamos o seu modelo treinado para checar se o sinal deve inverter
    bert_result = bert_scores[0]
    
    label_ia = bert_result['label']
    certeza_ia = bert_result['score']

    # Traduz o modelo de 3 classes para a mesma "língua" do seu algoritmo
    if label_ia in ['Positivo', 'LABEL_2']:
        opiniao_ia = 'pos'
    elif label_ia in ['Negativo', 'LABEL_0']:
        opiniao_ia = 'neg'
    else:
        opiniao_ia = 'neu'

    # ==========================================
    # 2. FLUXOGRAMA DE DECISÃO HÍBRIDA
    # ==========================================
    
    if p == 0:
        # --- CAMINHO ESQUERDO: Algoritmo não identificou nada ---
        
        # Caixa: "Modelo tem certeza MAIOR que 80% que é Posit ou Negat"
        if opiniao_ia in ['pos', 'neg'] and certeza_ia > 0.80:
            p = 2
            s = opiniao_ia # Aplica 'neg' ou 'pos' dependendo do modelo
        
        # Caixa tracejada azul: "Menor que 80% OU +80% que é NEUTRO"
        else:
            p = 0
            s = 'neu'

    else:
        # --- CAMINHO DIREITO: Algoritmo identificou algo (P != 0) ---
        
        if opiniao_ia == s:
            # Caixa: "Modelo CONCORDA em qualquer nível"
            # Mantém o peso de P e a direção intactos
            pass 
            
        else:
            # Modelo DISCORDA do Algoritmo
            
            # Caixa tracejada verde: "Discorda com mais de 80% de certeza que é POSITIVO ou NEGATIVO"
            if opiniao_ia != 'neu' and certeza_ia > 0.80:
                # Aqui ele classifica sarcasmo/inversão!
                # Como a IA tem +80% de certeza, ela vence. Trocamos o sentido (S).
                s = opiniao_ia 
                
            # Caixa tracejada azul direita: "Discorda com menos de 80% OU +80% que é NEUTRO"
            else:
                p = 0
                s = 'neu'

    # ==========================================
    # 3. RESULTADO FINAL (Ordem de Aprovação)
    # ==========================================
    # Agora sim aplicamos a regra matemática do diagrama:
    # Se s='pos', P fica positivo. Se s='neg', P fica negativo.
    p_matematico = p if s == 'pos' else (-p if s == 'neg' else 0)

    # --- NORMALIZAÇÃO (Escala 0.0 Bom a 1.0 Ruim) ---
    # Atenção: Como você retirou o limite de teto das palavras, seu P agora pode 
    # ficar maior que 7. Aumentei o divisor (14.0 para 20.0) para acomodar intensidades 
    # maiores sem o cálculo estourar o limite de 0 ou 1 logo de cara.
    normalized = 0.5 - (p_matematico / 20.0) 
    
    return max(0.0, min(1.0, round(normalized, 4)))