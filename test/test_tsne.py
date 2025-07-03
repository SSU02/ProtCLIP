# tests/test_tsne.py
import unittest
import torch
from utils import compute_tsne

class TestTSNE(unittest.TestCase):
    """
    Testing compute_tsne function
    """
    def test_tsne_output_shape(self):
        emb1 = torch.randn(1, 256)
        emb2 = torch.randn(1, 256)
        tsne_result = compute_tsne([emb1, emb2])
        self.assertEqual(tsne_result.shape, (2, 2))
