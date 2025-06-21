# visualize_embeddings.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer


texts = [
    # original
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
] + [
    # T5
    "Today is our dragon boat festival in our Chinese culture, to celebrate it safely and wonderfully in our lives. I hope you enjoy it too with my warmest wishes.",
    "In our Chinese culture, we celebrate the dragon boat festival today, wishing everyone safety and joy in life. I also hope you enjoy it as I do.",
    "Our Chinese dragon boat festival takes place today to celebrate safety and prosperity in our lives. My deepest wish is that you enjoy it as well.",
    # Pegasus
    "Happy Chinese New Year!",
    "Happy Chinese New Year to all!",
    "Happy Chinese New Year to you all!",
    # BART
    "Today is our dragon boat festival, in our Chinese culture to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
    "Today is our dragon boat festival in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes.",
    "Today is our dragon boat festival in Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes."
]

labels = (
    ["Original"] * 1 +
    ["T5"] * 3 +
    ["Pegasus"] * 3 +
    ["BART"] * 3
)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeds = model.encode(texts, convert_to_numpy=True)

pca = PCA(n_components=2)
emb_pca = pca.fit_transform(embeds)

tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=5)
emb_tsne = tsne.fit_transform(embeds)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for ax, data, title in zip(
        [ax1, ax2],
        [emb_pca, emb_tsne],
        ["PCA", "t-SNE (perplexity=5)"]
    ):
    for lbl in set(labels):
        pts = data[[i for i, l in enumerate(labels) if l == lbl]]
        ax.scatter(pts[:,0], pts[:,1], label=lbl, s=50)
    ax.set_title(title)
    ax.legend(loc='best', fontsize='small')

plt.tight_layout()
plt.show()
