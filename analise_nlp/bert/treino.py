import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

def treinar_modelo():
    print(f"CUDA disponível? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Placa detectada: {torch.cuda.get_device_name(0)}")

    # 1. Carregar o cérebro base (BERTimbau)
    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # num_labels=3 geralmente significa: 0 (Negativo), 1 (Neutro) e 2 (Positivo)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 2. Carregar o Dataset de Análise de Sentimento
    print("Baixando dataset jvanz/portuguese_sentiment_analysis...")
    dataset = load_dataset("jvanz/portuguese_sentiment_analysis")
    
    # O dataset será dividido: 80% para treino, 20% para a prova final (teste)
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    base_treino = dataset['train']
    base_teste = dataset['test']

    # 3. Função para converter o texto em números (Tokenização)
    def tokenize_function(examples):
        # Transforma o texto (usando a coluna padrão 'text')
        tokens = tokenizer(
            examples["review_text_processed"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        # Associa os gabaritos usando a coluna padrão 'label'
        tokens["labels"] = [int(label) for label in examples["polarity"]]
        
        return tokens

    print("Tokenizando os textos...")
    tokenized_train = base_treino.map(tokenize_function, batched=True)
    tokenized_test = base_teste.map(tokenize_function, batched=True)

    # 4. Configurações Específicas para a RTX (8GB VRAM)
    training_args = TrainingArguments(
        output_dir="./resultados_sentimento",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16, # Lote de textos por vez (seguro para 8GB)
        per_device_eval_batch_size=16,
        num_train_epochs=3,             # Vai ler a base inteira 3 vezes
        weight_decay=0.01,
        fp16=False,                      # O SEGREDO: Treinamento em meia-precisão (Rápido e leve)
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    # 5. Iniciar a "Faculdade"
    print("Iniciando o treinamento na GPU...")
    trainer.train()

    # 6. Salvar o diploma (Pesos do modelo treinado)
    pasta_destino = "./meu_bert_sentimento"
    model.save_pretrained(pasta_destino)
    tokenizer.save_pretrained(pasta_destino)
    print(f"Sucesso! Modelo salvo na pasta: {pasta_destino}")

if __name__ == "__main__":
    treinar_modelo()