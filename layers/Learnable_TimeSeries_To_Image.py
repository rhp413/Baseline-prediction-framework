import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTimeSeriesToImage(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)

    def forward(self, x_enc):
        B, L, D = x_enc.shape

        time_steps = torch.arange(L, dtype=torch.float32, device=x_enc.device).unsqueeze(0).repeat(B, 1)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)

        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)

        x_enc = x_enc.unsqueeze(-1)
        x_enc = torch.cat([x_enc, x_fft_mag, periodicity_encoding], dim=-1)
        x_enc = x_enc.permute(0, 2, 3, 1).reshape(B * D, 4, L)
        x_enc = self.conv1d(x_enc)
        x_enc = x_enc.reshape(B, D, self.hidden_dim, L).permute(0, 2, 1, 3)
        x_enc = torch.tanh(self.conv2d_1(x_enc))
        x_enc = torch.tanh(self.conv2d_2(x_enc))
        return F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
