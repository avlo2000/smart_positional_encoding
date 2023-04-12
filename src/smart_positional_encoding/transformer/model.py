import math

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransformerModel(nn.Module):

    def __init__(self,
                 pos_encoder,
                 n_tokens: int,
                 d_model: int,
                 n_heads: int,
                 d_hid: int,
                 n_layers: int,
                 dropout: float = 0.5
                 ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = pos_encoder
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_tokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
