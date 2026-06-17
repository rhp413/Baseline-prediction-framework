import einops
import torch.nn.functional as F


def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    B, seq_len, nvars = x_enc.shape

    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)
    images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)
    return images.mean(dim=1)
