# bart_paraphrase.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

texts = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"
]

def load_model():
    model_name = "eugenesiow/bart-paraphrase"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def paraphrase(text, tokenizer, model, device):
    inputs = tokenizer([text], max_length=256, truncation=True, padding="longest", return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
        num_return_sequences=3,
        repetition_penalty=1.2,
        length_penalty=0.8,
        early_stopping=True
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def main():
    tokenizer, model, device = load_model()
    for idx, txt in enumerate(texts, 1):
        print(f"\n Text {idx} ")
        paras = paraphrase(txt, tokenizer, model, device)
        for i, p in enumerate(paras, 1):
            print(f"Paraphrase {i}: {p}")

if __name__ == "__main__":
    main()
