import streamlit as st
import torch
from model import generate_embeddings, ESM2_Base_Model, ProtCLIP_ESM2_Model
from utils import compute_tsne
import plotly.express as px

st.title("ProtCLIP vs ESM2: Embedding Visualizer")

sequence = st.text_input("Enter your protein sequence:").upper()

if sequence:
    tokenizer, esm2_model, esm2_proj = ESM2_Base_Model.load_model()
    _, _, esm2_proj = ProtCLIP_ESM2_Model.load_model()

    esm2_emb = generate_embeddings(sequence, tokenizer, esm2_model, esm2_proj)
    protclip_emb = generate_embeddings(sequence, tokenizer, esm2_model, esm2_proj)

    # Run t-SNE
    tsne_data = compute_tsne([esm2_emb, protclip_emb])
    labels = ["ESM2 Base", "ProtCLIP-ESM2"]

    # Plot
    fig = px.scatter(x=tsne_data[:, 0], y=tsne_data[:, 1], text=labels, title="t-SNE Embedding Projection")
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig)
