import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("./sarcasm_distilbert")
model = DistilBertForSequenceClassification.from_pretrained("./sarcasm_distilbert")
model.eval()

st.title("Sarcasm Detector")

text = st.text_area("Enter a sentence:")

if st.button("Predict"):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    label = "Sarcastic" if probs[1] > probs[0] else "Not Sarcastic"
    confidence = probs.max().item()

    st.subheader(label)
    st.write(f"Confidence: {confidence:.2f}")