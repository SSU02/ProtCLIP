from sklearn.manifold import TSNE
import torch
import numpy as np

def compute_tsne(embeddings: list[torch.Tensor], perplexity: int = 1, max_iter: int = 1000):
    """
    Run t-SNE on a list of PyTorch embeddings.
    """
    embs = [e.cpu().numpy().flatten() for e in embeddings]
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42)
    tsne_result = tsne.fit_transform(np.array(embs))
    return tsne_result
