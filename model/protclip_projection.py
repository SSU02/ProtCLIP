import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv1d(nn.Module):
    """
    Applies a 1D convolution over an input sequence, preserving the input's 
    temporal structure by reshaping to match Conv1D's expected input format.
    
    This is used internally in attention pooling.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides. Default is 0.
        dilation (int, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of blocked connections. Default is 1.
        bias (bool, optional): If True, adds a learnable bias. Default is True.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, out_channels]
        """
        x = x.transpose(1, 2)  # [B, H, L]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, L, H']
        return x

class Attention1dPooling(nn.Module):
    """
    Applies attention-based pooling over a sequence of embeddings.
    
    Learns an attention score per token, and uses softmax to pool the weighted
    sum of hidden representations.
    
    Args:
        hidden_size (int): Size of the hidden embeddings.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Binary mask [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Pooled tensor [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        attn = self.layer(x).view(batch_size, -1)  # [B, L]

        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.bool(), float('-inf'))

        attn = F.softmax(attn, dim=-1).unsqueeze(-1)  # [B, L, 1]
        pooled = (attn * x).sum(dim=1)  # [B, H]
        return pooled


class ProjectionHead(nn.Module):
    """
    A multi-layer projection head that optionally applies attention pooling
    before reducing the input dimensionality for contrastive or embedding tasks.
    
    Args:
        input_dim (int): Input feature dimension.
        hidden_dim (int): Size of hidden layers in the projection. Default is 768.
        output_dim (int): Final projected dimension. Default is 256.
        dropout (float): Dropout probability between layers. Default is 0.2.
        use_attention (bool): Whether to apply attention pooling before projection.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        output_dim: int = 256,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.attention_pooling = Attention1dPooling(input_dim)

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, either [B, D] or [B, L, D]
            attention_mask (torch.Tensor, optional): Attention mask if using sequence input.
        
        Returns:
            torch.Tensor: Normalized output embedding [B, output_dim]
        """
        if self.use_attention and x.ndim == 3:
            x = self.attention_pooling(x, attention_mask)
        
        return F.normalize(self.projection(x), dim=-1)
