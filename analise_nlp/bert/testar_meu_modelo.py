from transformers import pipeline

caminho_do_seu_modelo = "./analise_3_classes_peso_vacina"

print("🧠 Carregando o SEU modelo treinado (Aguarde...)...\n")
classificador = pipeline("text-classification", model=caminho_do_seu_modelo)

# Dicionário de testes separados por categoria
bateria_de_testes = {
    " 1. Testes": [
        "Chata pqp",
        "Esse quadro atras de VC é  maravilhoso. E vc ta magnífica 🇧🇷🇧🇷",
        "Eu gravo novelinhas siga meu perfil pra ver",
        "Eu gravo novelinhas siga meu perfil pra ver espero que gostem",
        "Eu gravo novelinhas siga meu perfil pra ver espero que gostem",
        "Falo bonito , traduzindo ... ficou com medo",
        "Falou tudo gata 👏👏👏",
        "O Brasil tem que sentir orgulho de vc jordana, vc representa milhões de mulheres brasileiras, guereiras que estão na luta todos os dias!!!",
        "Oii eu gravo novelinhas siga meu perfil pra ver espero que gostem",
        "Para de mentira edu becks",
        "Vc faz joguinho tbm",
        "Vc.Jojo do passado,chamou Malévola pra briga kkkk e vc.Jojo de hoje puxou a orelha da Jojo do passado...ou foi sua equipe?kkkk",
        "Vender bet edifica a vida de quem ?",
    ]
    #"🟢 1. Textos Inofensivos (Fáceis)": [
    #    "Que dia maravilhoso, adorei a postagem!",
    #    "Concordo com você, excelente ponto de vista.",
    #    "Não entendi muito bem, pode explicar de novo?",
    #],
    #
    #"🔴 2. Ataques Diretos (Óbvio que é Tóxico)": [
    #    "Você é um lixo de ser humano, eu te odeio.",
    #    "Volta pro buraco de onde você saiu, seu imbecil.",
    #    "Tinha que ser muito burro pra falar uma merda dessas.",
    #],
#
    #"🟡 3. Crítica Dura (Não é crime/ódio, deveria ser Normal)": [
    #    "Achei o seu serviço péssimo, não recomendo para ninguém.",
    #    "Você não entende absolutamente nada sobre política ou economia.",
    #    "Isso é uma vergonha para o nosso país, atitude ridícula.",
    #],
#
    #"🟠 4. Falsos Positivos (Palavrão usado como elogio/ênfase)": [
    #    "Caralho, mano! Essa música nova ficou foda demais!",
    #    "Puta que pariu, que filme sensacional!",
    #    "Você é muito pica no que faz, parabéns!",
    #],
#
    #"🟣 5. Ameaças Sutis e Ironia (Difícil para IA)": [
    #    "Cuidado quando for andar na rua à noite, viu?",
    #    "Pena que não apanhou mais, merecia ter se machucado.",
    #    "Você deveria fazer um favor ao mundo e sumir.",
    #]
}

print("="*50)
print(" 🚀 INICIANDO BATERIA DE TESTES DO MODELO")
print("="*50)

for categoria, frases in bateria_de_testes.items():
    print(f"\n{categoria}")
    print("-" * 40)
    
    for texto in frases:
        resultado = classificador(texto)[0]
        label_ia = resultado['label']
        print(f"[DEBUG] Resultado bruto do modelo: {resultado}")
        print(f"[DEBUG] Label interpretada: {label_ia}")
        # Mapeamento do novo modelo: 0 (Negativo), 1 (Neutro), 2 (Positivo)
        # Verificamos tanto o nome amigável quanto o padrão (caso o id2label falhe)
        if label_ia in ['Negativo', 'LABEL_0']:
            classificacao = "🤬 NEGATIVO"
        elif label_ia in ['Neutro', 'LABEL_1']:
            classificacao = "😐 NEUTRO"
        elif label_ia in ['Positivo', 'LABEL_2']:
            classificacao = "✅ POSITIVO"
        else:
            classificacao = f"❓ {label_ia}"
            
        confianca = resultado['score'] * 100
        
        print(f"Texto: '{texto}'")
        print(f"IA: {classificacao} (Certeza: {confianca:.1f}%)\n")

print("="*50)
print("TESTES CONCLUÍDOS!")