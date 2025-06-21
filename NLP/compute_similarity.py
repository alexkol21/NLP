# compute_similarity.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

original_texts = [
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?"
]
paraphrases = {
    "T5": [
        "Today is our dragon boat festival in our Chinese culture to celebrate it safely and wonderfully in our lives. I hope you enjoy it too with my warmest wishes.",
        "In our Chinese culture, we celebrate the dragon boat festival today, wishing everyone safety and joy in life. I also hope you enjoy it as I do.",
        "Our Chinese dragon boat festival takes place today to celebrate safety and prosperity in our lives. My deepest wish is that you enjoy it as well."
    ],
    "Pegasus": [
        "Happy Chinese New Year!",
        "Happy Chinese New Year to all!",
        "Happy Chinese New Year to you all!"
    ],
    "BART": [
        "Today is our dragon boat festival, in our Chinese culture to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
        "Today is our dragon boat festival in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
        "Today is our dragon boat festival in Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes."
    ]
}

print("Φορτωση SentenceTransformer model")
model = SentenceTransformer('all-MiniLM-L6-v2')


print("Υπολογισμος embeddings για originals")
orig_embeds = model.encode(original_texts, convert_to_tensor=True)

records = []
for pipe, paras in paraphrases.items():
    for i, para in enumerate(paras, start=1):
        sim = util.cos_sim(orig_embeds[0], model.encode(para, convert_to_tensor=True)).item()
        records.append({
            "Pipeline": pipe,
            "Original_ID": 1,
            "Paraphrase_ID": i,
            "Cosine_Similarity": sim
        })

df = pd.DataFrame.from_records(records)
print("\nCosine Similarities")
print(df)
df.to_csv("similarities.csv", index=False)
print("\nSaved similarities.csv")
