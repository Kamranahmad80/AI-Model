# File: models/bert_embedding.py
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
from transformers import BertTokenizer, BertModel
import torch

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', use_safetensors=False)

def get_bert_embedding(text):
    """
    Given a text string, return its BERT embedding.
    We use mean pooling over token embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Average the token embeddings to form a single vector representation
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

if __name__ == "__main__":
    # Example usage
    sample_text = "Software engineer with experience in machine learning and data science."
    emb = get_bert_embedding(sample_text)
    print("Embedding shape:", emb.shape)


