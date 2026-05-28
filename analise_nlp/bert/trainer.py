import torch
from torch import nn
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def treinar_modelo_multiclasse():
    print(f"CUDA disponível? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Placa detectada: {torch.cuda.get_device_name(0)}")

    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Mapeamento para o modelo saber os nomes das 3 classes
    id2label = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
    label2id = {"Negativo": 0, "Neutro": 1, "Positivo": 2}

    # Alterado para num_labels=3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # ==========================================
    # 1. PREPARANDO O TOLD-BR -> CLASSE 0 (NEGATIVO)
    # ==========================================
    print("Baixando, filtrando e harmonizando ToLD-Br...")
    toldbr = load_dataset("told-br", trust_remote_code=True)
    toldbr = toldbr['train']
    
    def is_toxic(example):
        for coluna, valor in example.items():
            if coluna != "text":
                try:
                    if float(valor) > 0.0:
                        return True
                except (ValueError, TypeError):
                    continue
        return False

    # Filtrar apenas as linhas tóxicas
    toldbr_ofensivo = toldbr.filter(is_toxic)
    
    # Formatar: label como 0 (Negativo)
    def formatar_toldbr_ofensivo(example):
        return {
            "text": str(example["text"]),
            "labels": 0
        }
        
    toldbr_formatado = toldbr_ofensivo.map(formatar_toldbr_ofensivo, remove_columns=toldbr_ofensivo.column_names)

    # ==========================================
    # 2. PREPARANDO O SEU CSV -> CLASSES 1 E 2 (NEUTRO E POSITIVO)
    # ==========================================
    print("Carregando o CSV personalizado (Dataset Modelo Final)...")
    
    caminho_csv = "./dataset_modelo_final.csv"
    df_custom = pd.read_csv(caminho_csv)
    
    # Filtrar para garantir que só temos Neutral e Positive
    df_custom = df_custom[df_custom['HandLabel'].isin(['Positive', 'Neutral','Negative'])]
    
    # Função para mapear o texto do HandLabel para os IDs numéricos
    def mapear_labels_csv(label):
        if label == 'Neutral':
            return 1 # Neutro
        elif label == 'Positive':
            return 2 # Positivo
        elif label == 'Negative':
            return 0 # Negativo
        return -1
    
    df_custom['labels'] = df_custom['HandLabel'].apply(mapear_labels_csv)
    
    # Manter apenas as colunas necessárias
    df_custom = df_custom[['text', 'labels']]
    
    csv_formatado = Dataset.from_pandas(df_custom)
    if '__index_level_0__' in csv_formatado.column_names:
        csv_formatado = csv_formatado.remove_columns(['__index_level_0__'])

    # ==========================================
    # 2.5 INJEÇÃO DE "VACINA" CONTRA VIÉS LEXICAL
    # ==========================================
    print("Injetando vacina contra viés de palavrões...")
    
    frases_positivas_vacina = [
        "Caralho, que filme bom!", "Essa música é foda demais, não consigo parar de ouvir.", "Puta merda, você arrasou nessa apresentação!", "Mano, tu é muito pica no que faz, parabéns.", "Que comida gostosa pra porra!", "Porra, finalmente consegui passar na prova!", "Puta que pariu, que vista incrível desse lugar.", "Esse jogo novo tá do caralho.", "Caralho, mano, tô muito feliz por você!", "Foda demais ver você crescendo na carreira.", "O show ontem foi pica pra caralho!", "Meu Deus, que bolo bom da porra.", "Pqp, melhor notícia do ano!", "Você é foda, obrigado por sempre me ajudar.", "Cacete, que gol lindo!", "Mano do céu, que camisa foda!", "Puta merda, eu amo muito essa banda.", "Porra, esse livro mudou minha vida.", "Caralho, que orgulho de você!", "Tá muito foda esse teu projeto novo.", "Puta que o pariu, que rolê inesquecível!", "A festa de ontem foi foda, me diverti muito.", "Você cozinha bem pra caralho.", "Porra, tu é um gênio, mano!", "Que ideia do caralho, vamos fazer isso sim.", "Cacete, você desenha muito bem!", "Mano, teu cabelo ficou foda assim.", "Pqp, eu ri pra caralho com esse vídeo.", "Caralho, que sensação maravilhosa estar aqui.", "Essa sua atitude foi muito pica.", "Puta merda, que saudade que eu tava de você!", "Porra, que carro lindo.", "Ficou foda demais a nova decoração da casa.", "Caralho, você canta muito!", "Pqp, tô feliz pra porra hoje!", "O atendimento desse lugar é foda, muito prestativos.", "Que viagem do caralho, quero voltar logo.", "Porra, acertou em cheio no presente!", "Você é o cara mais foda que eu conheço.", "Cacete, que evolução fantástica a sua."
    ]

    frases_neutras_vacina = [
        "Eita porra, esqueci a carteira em casa.", "Caralho, que susto você me deu!", "Puta merda, choveu bem na hora de sair.", "Que frio do caralho hoje.", "Foda-se, não vou me estressar com isso não.", "Cacete, bati o dedinho na quina da mesa.", "Porra, o ônibus atrasou de novo.", "Pqp, deixei o café esfriar.", "Mano, tá um calor da porra hoje.", "Caralho, já são seis da tarde e não terminei meu trabalho."
    ]

    # ---> SOLUÇÃO DO ERRO: Declarar o features_padrao AQUI, antes do cast! <---
    features_padrao = Features({
        "text": Value("string"),
        "labels": Value("int64")
    })
    
    # Criando e tipando o dataset Positivo
    vacina_positiva = Dataset.from_dict({
        "text": frases_positivas_vacina,
        "labels": [2] * len(frases_positivas_vacina)
    }).cast(features_padrao)
    
    # Criando e tipando o dataset Neutro
    vacina_neutra = Dataset.from_dict({
        "text": frases_neutras_vacina,
        "labels": [1] * len(frases_neutras_vacina)
    }).cast(features_padrao)

    # ==========================================
    # 3. CONCATENAÇÃO E TOKENIZAÇÃO
    # ==========================================
    print("Padronizando os tipos das colunas antes de juntar...")

    # Forçar (cast) os datasets principais a usarem exatamente o mesmo tipo
    toldbr_formatado = toldbr_formatado.cast(features_padrao)
    csv_formatado = csv_formatado.cast(features_padrao)

    print("Juntando todos os datasets (agora com a vacina!)...")
    # Adicionando as vacinas na lista de concatenação
    dataset_completo = concatenate_datasets([toldbr_formatado, csv_formatado, vacina_positiva, vacina_neutra])
    
    dataset_completo = dataset_completo.shuffle(seed=42)
    split_dataset = dataset_completo.train_test_split(test_size=0.2, seed=42)

    treino_final = split_dataset['train']
    teste_final = split_dataset['test']

    def tokenizar_tudo(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    treino_tokenizado = treino_final.map(tokenizar_tudo, batched=True)
    teste_tokenizado = teste_final.map(tokenizar_tudo, batched=True)

    # ==========================================
    # 4. TREINAMENTO COM PESOS (CUSTOM TRAINER)
    # ==========================================
    print("Configurando pesos das classes e o Custom Trainer...")
    
    # 1. Definição dos pesos (Exemplo temporário, ajustaremos os valores reais depois)
    # Negativo (0), Neutro (1), Positivo (2)
    class_weights = torch.tensor([0.83, 1.11, 1.12]) 
    
    # Enviar os pesos para a Placa de Vídeo (CRÍTICO para não dar erro)
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()

    # 2. Criação do "Professor Rigoroso" (Custom Trainer)
    class CustomTrainer(Trainer):
        # O **kwargs previne erros em versões mais novas do Hugging Face
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Tira os gabaritos (labels) dos inputs
            labels = inputs.pop("labels")
            
            # Pede para o modelo tentar adivinhar
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Envia os pesos para o mesmo dispositivo onde o modelo está (CPU ou GPU)
            pesos = class_weights.to(model.device)
            
            # Aplica a punição severa
            loss_fct = nn.CrossEntropyLoss(weight=pesos)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss

    # 3. Argumentos de Treino
    training_args = TrainingArguments(
        output_dir="./resultados_modelo_3_classes",
        evaluation_strategy="epoch", # <-- VOLTE PARA ESTE NOME
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True, 
        save_strategy="epoch",
    )

    # 4. Instanciando o SEU CustomTrainer em vez do Trainer normal
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=treino_tokenizado,
        eval_dataset=teste_tokenizado,
    )

    print("Iniciando o Treinamento Multiclasse (Negativo / Neutro / Positivo)...")
    trainer.train()

    # 6. Salvar o modelo (Mantido igual)
    pasta_destino = "./analise_3_classes_peso_vacina"
    model.save_pretrained(pasta_destino)
    tokenizer.save_pretrained(pasta_destino)
    print(f"🚀 SUCESSO! Modelo salvo em: {pasta_destino}")

if __name__ == "__main__":
    treinar_modelo_multiclasse()