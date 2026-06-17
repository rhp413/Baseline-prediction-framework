import torch

from transformers import CLIPModel, CLIPProcessor


class VLMManager:
    def __init__(self, config):
        self.config = config
        self.vlm_type = config.vlm_type.lower()
        if self.vlm_type != 'clip':
            raise ValueError('This cleaned benchmark keeps only the CLIP Time-VLM branch.')

        self.device = self._acquire_device()
        self._init_clip()
        self.model.to(self.device)
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("VLM Learnable model parameters: {:,}".format(learnable_params))

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{self.config.gpu}')
        print('Use CPU')
        return torch.device('cpu')

    def _init_clip(self):
        clip_path = '/root/phr/ICML25-TimeVLM-main/clip-vit-base-patch32'
        print(f"[VLMManager] Loading CLIP from local path: {clip_path}")
        self.processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(clip_path, local_files_only=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.max_input_text_length = 77

    def _set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value

    def process_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        return outputs.image_embeds, outputs.text_embeds
