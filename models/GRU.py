import torch
import torch.nn as nn


class Model(nn.Module):
    """
    GRU baseline for long-term forecasting.

    input:
        x_enc: [B, seq_len, enc_in]
    output:
        [B, pred_len, c_out]
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.hidden_size = getattr(configs, "gru_hidden_size", configs.d_model)
        self.num_layers = getattr(configs, "gru_layers", configs.e_layers)
        self.dropout = getattr(configs, "dropout", 0.1)
        self.bidirectional = getattr(configs, "gru_bidirectional", False)
        self.output_attention = getattr(configs, "output_attention", False)

        gru_dropout = self.dropout if self.num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=self.bidirectional
        )

        out_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(out_dim, configs.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(configs.d_ff, self.pred_len * self.c_out)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        batch_size = x_enc.size(0)

        _, h_n = self.gru(x_enc)

        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]

        out = self.projection(last_hidden)
        out = out.view(batch_size, self.pred_len, self.c_out)

        if self.output_attention:
            return out, None
        return out