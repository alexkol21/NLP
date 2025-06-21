# t5_paraphrase.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes."""
text2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"""


model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def paraphrase_text(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(input_text, padding="max_length", return_tensors="pt", max_length=128, truncation=True)
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=5,
        num_return_sequences=3,
        temperature=1.5
    )

    print(f"\nOriginal: {text}\n")
    for i, output in enumerate(outputs):
        paraphrased = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"Paraphrase {i+1}: {paraphrased}")


paraphrase_text(text1)
paraphrase_text(text2)
