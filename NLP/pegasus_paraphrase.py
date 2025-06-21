# pegasus_paraphrase.py

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


texts = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"
]

def load_model():
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def paraphrase(text: str, tokenizer, model, device):
    batch = tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
    outputs = model.generate(
        **batch,
        max_length=128,
        num_beams=8,
        num_return_sequences=3,
        length_penalty=0.9,
        early_stopping=True
    )
    # Αποκωδικοποίηση
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

def main():
    tokenizer, model, device = load_model()
    for idx, txt in enumerate(texts, 1):
        print(f"\n--- Text {idx} ---")
        paras = paraphrase(txt, tokenizer, model, device)
        for i, p in enumerate(paras, 1):
            print(f"Paraphrase {i}: {p}")

if __name__ == "__main__":
    main()
