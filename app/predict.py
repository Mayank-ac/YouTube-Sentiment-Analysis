import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load BERT-based sentiment analysis model
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def encode_texts(text_list):
    """Encodes text using the BERT tokenizer."""
    encodings = tokenizer(text_list, truncation=True, padding=True, max_length=512, return_tensors="pt")
    return encodings.to(device)  # Move to GPU if available

def predict_sentiments(text_list):
    """Predicts sentiment for a list of texts using BERT and maps it to Flask-compatible labels."""
    encoded_inputs = encode_texts(text_list)

    with torch.no_grad():
        outputs = model(**encoded_inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy() + 1  # Convert to 1-5 scale

    # Convert BERT's 1-5 scale to "Positive" or "Negative"
    sentiment_map = {
        1: "Negative",   # Very Negative
        2: "Negative",   # Negative
        3: "Neutral",    # Neutral (Ignored in Flask)
        4: "Positive",   # Positive
        5: "Positive"    # Very Positive
    }
    
    return [sentiment_map[pred] for pred in predictions]

