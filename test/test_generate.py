import unittest
from model import generate_embeddings, ESM2_Base_Model, ProtCLIP_ESM2_Model

class TestGenerateEmbeddings(unittest.TestCase):
    def test_generate_embeddings_for_esm2_base_model(self):
        """
        Test generating of embeddings for the base ESM2 model and tokenizer.
        """
        tokenizer, model, esm2_proj = ESM2_Base_Model.load_model()
        sequence = "SRI"
        embeddings = generate_embeddings(sequence, tokenizer, model, esm2_proj)
        self.assertIsNotNone(embeddings)

    def test_generate_embeddings_for_protclip_esm2_model(self):
        """
        Test generating of embeddings for ProtCLIP ESM2 model and tokenizer.
        """
        tokenizer, model, esm2_proj = ProtCLIP_ESM2_Model.load_model()
        sequence = "SRI"
        embeddings = generate_embeddings(sequence, tokenizer, model, esm2_proj)
        self.assertIsNotNone(embeddings)