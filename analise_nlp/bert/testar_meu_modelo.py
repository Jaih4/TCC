from transformers import pipeline

caminho_do_seu_modelo = "./bert_sota_toxicidade_v2"

print("🧠 Carregando o SEU modelo treinado (Aguarde...)...\n")
classificador = pipeline("text-classification", model=caminho_do_seu_modelo)

# Dicionário de testes separados por categoria
bateria_de_testes = {
    "🟢 1. Textos Inofensivos (Fáceis)": [
        "Que dia maravilhoso, adorei a postagem!",
        "Concordo com você, excelente ponto de vista.",
        "Não entendi muito bem, pode explicar de novo?",
    ],
    
    "🔴 2. Ataques Diretos (Óbvio que é Tóxico)": [
        "Você é um lixo de ser humano, eu te odeio.",
        "Volta pro buraco de onde você saiu, seu imbecil.",
        "Tinha que ser muito burro pra falar uma merda dessas.",
    ],

    "🟡 3. Crítica Dura (Não é crime/ódio, deveria ser Normal)": [
        "Achei o seu serviço péssimo, não recomendo para ninguém.",
        "Você não entende absolutamente nada sobre política ou economia.",
        "Isso é uma vergonha para o nosso país, atitude ridícula.",
    ],

    "🟠 4. Falsos Positivos (Palavrão usado como elogio/ênfase)": [
        "Caralho, mano! Essa música nova ficou foda demais!",
        "Puta que pariu, que filme sensacional!",
        "Você é muito pica no que faz, parabéns!",
    ],

    "🟣 5. Ameaças Sutis e Ironia (Difícil para IA)": [
        "Cuidado quando for andar na rua à noite, viu?",
        "Pena que não apanhou mais, merecia ter se machucado.",
        "Você deveria fazer um favor ao mundo e sumir.",
    ]
}

print("="*50)
print(" 🚀 INICIANDO BATERIA DE TESTES DO MODELO")
print("="*50)

for categoria, frases in bateria_de_testes.items():
    print(f"\n{categoria}")
    print("-" * 40)
    
    for texto in frases:
        resultado = classificador(texto)[0]
        
        # O modelo HateBR foi treinado com 1 para Tóxico e 0 para Normal
        if resultado['label'] == 'LABEL_1':
            classificacao = "🤬 TÓXICO"
        else:
            classificacao = "✅ NORMAL"
            
        confianca = resultado['score'] * 100
        
        print(f"Texto: '{texto}'")
        print(f"IA: {classificacao} (Certeza: {confianca:.1f}%)\n")

print("="*50)
print("TESTES CONCLUÍDOS!")