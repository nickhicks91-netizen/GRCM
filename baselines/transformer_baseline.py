"""
Standard Transformer Baseline
Comparable architecture to GRCM for efficiency benchmarking
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class BaselineConfig:
    """Configuration for transformer baseline"""
    input_dim: int = 15
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 512
    dropout: float = 0.1
    output_dim: int = 15
    max_seq_len: int = 128
    device: str = 'cpu'


class TransformerBaseline(nn.Module):
    """
    Standard PyTorch Transformer for comparison with GRCM

    Architecture designed to be comparable to GRCM:
    - Similar parameter count
    - Similar input/output dimensions
    - Standard multihead attention (vs resonant attention)
    - Feed-forward layers (vs frequency-based processing)
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        super().__init__()
        self.config = config or BaselineConfig()

        # Input projection
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.config.max_seq_len, self.config.hidden_dim)
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor [batch, seq_len, input_dim] or [batch, input_dim]
            mask: Optional attention mask

        Returns:
            Dictionary with output and intermediate values
        """
        batch_size = x.size(0)

        # Handle 2D input (single timestep)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        seq_len = x.size(1)

        # Project input to hidden dimension
        h = self.input_proj(x)  # [batch, seq_len, hidden_dim]

        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len, :]

        # Apply transformer
        h = self.transformer(h, mask=mask)  # [batch, seq_len, hidden_dim]

        # Layer norm
        h = self.layer_norm(h)

        # Output projection
        output = self.output_proj(h)  # [batch, seq_len, output_dim]

        # Take last timestep for single output
        if seq_len == 1:
            output = output.squeeze(1)  # [batch, output_dim]
            h = h.squeeze(1)  # [batch, hidden_dim]

        return {
            'output': output,
            'hidden': h,
            'attention_weights': None  # Standard transformer doesn't expose these easily
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': 'TransformerBaseline',
            'num_parameters': self.count_parameters(),
            'num_layers': self.config.num_layers,
            'hidden_dim': self.config.hidden_dim,
            'num_heads': self.config.num_heads,
            'ff_dim': self.config.ff_dim
        }


class MultimodalTransformerBaseline(nn.Module):
    """
    Multimodal Transformer Baseline
    Matches GRCM's multimodal input structure more closely
    """

    def __init__(self, config: Optional[BaselineConfig] = None):
        super().__init__()
        self.config = config or BaselineConfig()

        # Modality-specific projections (matching GRCM's inputs)
        self.image_proj = nn.Linear(512, self.config.hidden_dim)  # CLIP embeddings
        self.audio_proj = nn.Linear(768, self.config.hidden_dim)  # Wav2Vec embeddings
        self.action_proj = nn.Linear(4, self.config.hidden_dim)   # Action vector

        # Modality fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            batch_first=True
        )

        # Main transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers
        )

        # Output heads (matching GRCM outputs)
        self.output_proj = nn.Linear(self.config.hidden_dim, 15)
        self.coherence_head = nn.Linear(self.config.hidden_dim, 1)
        self.qualia_head = nn.Linear(self.config.hidden_dim, 4)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        image_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multimodal inputs (matching GRCM signature)

        Args:
            image_emb: Image embeddings [batch, 512]
            audio_emb: Audio embeddings [batch, 768]
            action: Action vector [batch, 4]

        Returns:
            Dictionary with outputs (matching GRCM output format)
        """
        batch_size = image_emb.size(0)

        # Project modalities to common dimension
        img_h = self.image_proj(image_emb).unsqueeze(1)  # [batch, 1, hidden]
        aud_h = self.audio_proj(audio_emb).unsqueeze(1)  # [batch, 1, hidden]
        act_h = self.action_proj(action).unsqueeze(1)    # [batch, 1, hidden]

        # Concatenate modalities
        h = torch.cat([img_h, aud_h, act_h], dim=1)  # [batch, 3, hidden]

        # Multimodal fusion via self-attention
        h_fused, _ = self.fusion(h, h, h)  # [batch, 3, hidden]

        # Apply transformer
        h_out = self.transformer(h_fused)  # [batch, 3, hidden]

        # Pool across modalities (mean pooling)
        h_pooled = h_out.mean(dim=1)  # [batch, hidden]

        # Generate outputs
        output = self.output_proj(h_pooled)
        coherence = torch.sigmoid(self.coherence_head(h_pooled))
        qualia = torch.softmax(self.qualia_head(h_pooled), dim=-1)

        return {
            'output': output,
            'coherence': coherence,
            'qualia': qualia,
            'hidden': h_pooled,
            # Add dummy outputs to match GRCM interface
            'memory': torch.zeros(32, device=output.device),
            'reflection': torch.zeros(batch_size, device=output.device),
            'desire_align': torch.zeros(batch_size, 1, device=output.device),
            'phi': 0.0,
            'prop_state': torch.zeros(batch_size, 16, device=output.device),
            'identity_token': torch.zeros(32, device=output.device),
            'timestamp': 0,
            'ethical_status': {'halt': False, 'reason': None}
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': 'MultimodalTransformerBaseline',
            'num_parameters': self.count_parameters(),
            'num_layers': self.config.num_layers,
            'hidden_dim': self.config.hidden_dim,
            'num_heads': self.config.num_heads,
            'ff_dim': self.config.ff_dim
        }


def create_comparable_baseline(grcm_config) -> MultimodalTransformerBaseline:
    """
    Create a transformer baseline with comparable complexity to GRCM

    Args:
        grcm_config: GRCM configuration to match

    Returns:
        MultimodalTransformerBaseline with similar parameter count
    """
    # Match hidden dimension to GRCM's frequency dimension
    baseline_config = BaselineConfig(
        input_dim=grcm_config.input_dim,
        hidden_dim=grcm_config.freq_dim * 16,  # Scale up to match GRCM complexity
        num_heads=4,
        num_layers=3,
        ff_dim=grcm_config.freq_dim * 64,
        dropout=0.1,
        output_dim=grcm_config.input_dim,
        device=grcm_config.device
    )

    return MultimodalTransformerBaseline(baseline_config)
