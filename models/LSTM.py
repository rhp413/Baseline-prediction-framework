import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Simple LSTM baseline for time series forecasting.

    Compatible with the project interface:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)

    Notes:
    - For forecasting tasks, this model uses only x_enc and ignores x_mark/x_dec.
    - It encodes the input sequence with LSTM, then maps the last hidden state
      to future prediction steps by a linear head.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, "label_len", 0)
        self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = max(1, configs.e_layers)

        # PyTorch LSTM dropout only works when num_layers > 1
        self.lstm_dropout = configs.dropout if self.num_layers > 1 else 0.0

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
            bidirectional=False
        )

        # Forecast head: last hidden -> pred_len * c_out
        self.forecast_head = nn.Linear(self.hidden_size, self.pred_len * self.c_out)

        # Shared projection for sequence-level tasks
        self.projection = nn.Linear(self.hidden_size, self.c_out)

        # Classification head
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.classifier = nn.Linear(self.hidden_size * self.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [B, seq_len, enc_in]
        return: [B, pred_len, c_out]
        """
        enc_out, (h_n, c_n) = self.encoder(x_enc)

        # h_n: [num_layers, B, hidden_size]
        last_hidden = h_n[-1]  # [B, hidden_size]

        dec_out = self.forecast_head(last_hidden)  # [B, pred_len * c_out]
        dec_out = dec_out.view(x_enc.size(0), self.pred_len, self.c_out)

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        Sequence-to-sequence style output for imputation.
        return: [B, seq_len, c_out]
        """
        enc_out, _ = self.encoder(x_enc)              # [B, seq_len, hidden_size]
        dec_out = self.projection(enc_out)            # [B, seq_len, c_out]
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        Sequence reconstruction-style output.
        return: [B, seq_len, c_out]
        """
        enc_out, _ = self.encoder(x_enc)              # [B, seq_len, hidden_size]
        dec_out = self.projection(enc_out)            # [B, seq_len, c_out]
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        return: [B, num_class]
        """
        enc_out, _ = self.encoder(x_enc)              # [B, seq_len, hidden_size]
        output = self.act(enc_out)
        output = self.dropout(output)

        # If x_mark_enc is a padding mask in classification task, keep compatibility
        if x_mark_enc is not None and x_mark_enc.dim() == 2 and x_mark_enc.shape[1] == output.shape[1]:
            output = output * x_mark_enc.unsqueeze(-1)

        output = output.reshape(output.shape[0], -1)  # [B, seq_len * hidden_size]
        output = self.classifier(output)              # [B, num_class]
        return output

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Keep same behavior as forecast for compatibility.
        """
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]

        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]

        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]

        return None 