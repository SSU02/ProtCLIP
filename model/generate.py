import torch

def generate_embeddings(sequence: str, tokenizer, model, esm2_proj=None):
    """
    Generate embeddings for the given sequence using 
    the loaded tokenizer, model, and optional projection head.

    Args:
        sequence (str): Protein or text sequence input.
        tokenizer: Tokenizer object to preprocess input.
        model: Pretrained model to extract embeddings.
        esm2_proj (optional): ProjectionHead model for embedding projection.

    Returns:
        torch.Tensor: Generated embedding tensor.
    """

    inputs = tokenizer(
        sequence, 
        return_tensors='pt',
        padding=True, 
        truncation=True, 
        max_length = 256
        )
    if esm2_proj:
        with torch.no_grad():
            outputs = model(**inputs)
            embs = outputs.last_hidden_state.mean(dim=1)
            embs = esm2_proj(embs)
            return embs

    else:
        with torch.no_grad():
            outputs = model(**inputs)
            embs = outputs.last_hidden_state.mean(dim=1)
            return embs