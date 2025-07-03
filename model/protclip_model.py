from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
from .protclip_projection import ProjectionHead
import torch

class ProtCLIP_ESM2_Model:
    """
    Wrapper class for the ProtCLIP ESM2 model.

    This class loads the pre-trained ESM2 model and tokenizer (650M) 
    and the ProtCLIP projection head from Hugging Face.
    """

    @staticmethod
    def load_model():
        """
        Load the ProtCLIP ESM2 tokenizer, model, and projection head.

        Returns:
            tokenizer : Pretrained ESM2 tokenizer.
            model     : Pretrained ESM2 model.
            esm2_proj : Loaded ProtCLIP projection head.
        """
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
        model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
        protclip_model_path = hf_hub_download(repo_id="SSU02/protclip", filename="protclip.pt")

        try:
            protclip_model = torch.load(protclip_model_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError("Failed to load ProtCLIP weights") from e
        
        esm2_proj = ProjectionHead(1280)
        esm2_proj_dict = {k.replace('esm2_proj.', ''): v for k, v in protclip_model.items() 
                             if 'esm2_proj' in k}
        
        esm2_proj.load_state_dict(esm2_proj_dict)

        return tokenizer, model, esm2_proj 