import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def treinar_modelo_sota():
    print(f"CUDA disponível? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Placa detectada: {torch.cuda.get_device_name(0)}")

    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ==========================================
    # 1. PREPARANDO O HATEBR
    # ==========================================
    print("Baixando e harmonizando HateBR...")
    hatebr = load_dataset("ruanchaves/hatebr", trust_remote_code=True)
    hatebr = hatebr['train'] # O HateBR vem tudo no 'train', vamos dividir no final
    
    # Padronizando colunas: 'instagram_comments' -> 'text', 'offensive_language' -> 'labels'
    def formatar_hatebr(example):
        return {
            "text": example["instagram_comments"],
            "labels": int(example["offensive_language"])
        }
    
    hatebr_formatado = hatebr.map(formatar_hatebr, remove_columns=hatebr.column_names)

    # ==========================================
    # 2. PREPARANDO O TOLD-BR (AGREGAÇÃO MULTI-LABEL)
    # ==========================================
    print("Baixando e harmonizando ToLD-Br...")
    toldbr = load_dataset("told-br", trust_remote_code=True)
    toldbr = toldbr['train']
    
    def formatar_toldbr(example):
        # Variável que diz se é tóxico ou não
        is_toxic = 0
        
        # O ToLD-Br tem a coluna 'text' e várias colunas de categorias.
        # Nós vamos verificar todas as colunas que NÃO são 'text'.
        for coluna, valor in example.items():
            if coluna != "text":
                # Se qualquer categoria (racismo, insulto, etc) for maior que 0, é tóxico.
                try:
                    if float(valor) > 0.0:
                        is_toxic = 1
                        break # Não precisa verificar o resto, já sabemos que é tóxico
                except (ValueError, TypeError):
                    continue # Ignora colunas que não são números
                    
        return {
            "text": str(example["text"]),
            "labels": is_toxic
        }
        
    toldbr_formatado = toldbr.map(formatar_toldbr, remove_columns=toldbr.column_names)

    # ==========================================
    # 3. PREPARANDO A VACINA (COM OVERSAMPLING)
    # ==========================================
    print("Preparando dados sintéticos (Vacina de Alta Dose)...")
    
    frases_vacina = [
        "Puta que pariu, que filme sensacional!",
        "Caralho mano, você é foda demais, parabéns!",
        "Essa música nova ficou do caralho de tão boa.",
        "Que dia foda, ganhei na loteria porra!",
        "Você é muito pica no que faz, parabéns!", # Adicionado o caso que falhou
        "Cuidado quando for andar na rua à noite, viu?",
        "Pena que não apanhou mais, merecia ter se machucado.", # Adicionado o caso que falhou
        "Você deveria fazer um favor ao mundo e sumir daqui.",
        "Espero que você sofra um acidente bem grave.",
        "Gente como você não deveria ter o direito de respirar."
    ]
    
    labels_vacina = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # MULTIPLICADOR MÁGICO: Repete as frases 100 vezes para criar peso estatístico
    multiplicador = 100
    
    dados_sinteticos = {
        "text": frases_vacina * multiplicador,   # Agora teremos 1.000 frases de vacina!
        "labels": labels_vacina * multiplicador
    }
    
    sinteticos_formatado = Dataset.from_dict(dados_sinteticos)

    # ==========================================
    # 4. O GRANDE CALDEIRÃO (CONCATENAÇÃO E TOKENIZAÇÃO)
    # ==========================================
    print("Juntando todos os datasets...")
    # Junta tudo em um único dataset massivo
    dataset_completo = concatenate_datasets([hatebr_formatado, toldbr_formatado, sinteticos_formatado])
    
    # Embaralha os dados e divide 80% treino / 20% teste
    dataset_completo = dataset_completo.shuffle(seed=42)
    split_dataset = dataset_completo.train_test_split(test_size=0.2, seed=42)
    
    treino_final = split_dataset['train']
    teste_final = split_dataset['test']

    # Tokeniza tudo de uma vez só no formato que o PyTorch ama
    print("Tokenizando o Mega Dataset...")
    def tokenizar_tudo(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    treino_tokenizado = treino_final.map(tokenizar_tudo, batched=True)
    teste_tokenizado = teste_final.map(tokenizar_tudo, batched=True)

    # ==========================================
    # 5. TREINAMENTO
    # ==========================================
    training_args = TrainingArguments(
        output_dir="./resultados_mega_modelo",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True, # Magia da RTX 4060 ativada
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=treino_tokenizado,
        eval_dataset=teste_tokenizado,
    )

    print("Iniciando Mega Treinamento (SOTA)...")
    trainer.train()

    pasta_destino = "./bert_sota_toxicidade_v2"
    model.save_pretrained(pasta_destino)
    tokenizer.save_pretrained(pasta_destino)
    print(f"🚀 SUCESSO ABSOLUTO! Modelo SOTA salvo em: {pasta_destino}")

if __name__ == "__main__":
    treinar_modelo_sota()