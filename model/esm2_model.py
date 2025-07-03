from transformers import AutoTokenizer, AutoModel
from .protclip_projection import ProjectionHead #new

class ESM2_Base_Model:
    """
    Wrapper class for the ESM-2 protein language model from Facebook.

    This class provides a convenient method to load the pre-trained 
    650M parameters variant tokenizer and model weights for the ESM2 model.
    """

    @staticmethod
    def load_model():
        """
        Load the pretrained ESM2 (650M parameter variant) model and tokenizer from Hugging Face.

        Returns:
            tokenizer : Pretrained ESM2 tokenizer.
            model     : Pretrained ESM2 model.
        """
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
        model = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
        esm2_proj = ProjectionHead(1280) #new
        
        # return tokenizer, model
        return tokenizer, model, esm2_proj #new
        