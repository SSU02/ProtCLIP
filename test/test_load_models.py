import unittest
from model import ESM2_Base_Model, ProtCLIP_ESM2_Model

class TestLoadModels(unittest.TestCase):
    def test_load_esm2_base_model(self):
        """
        Test loading of the base ESM2 model and tokenizer.
        """
        # tokenizer, model = ESM2_Base_Model.load_model()
        # self.assertIsNotNone(tokenizer)
        # self.assertIsNotNone(model)

        tokenizer, model, esm2_proj = ESM2_Base_Model.load_model()
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)
        self.assertIsNotNone(esm2_proj)
        
    def test_load_protclip_esm2_model(self):
        """
        Test loading of the ProtCLIP ESM2 model components.
        """
        tokenizer, model, esm2_proj = ProtCLIP_ESM2_Model.load_model()
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)
        self.assertIsNotNone(esm2_proj)