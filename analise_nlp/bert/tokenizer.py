from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize_text(text: str):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True)