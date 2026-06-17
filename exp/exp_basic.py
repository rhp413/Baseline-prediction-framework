import torch
from models import DLinear, GRU, Informer, LSTM, PatchTST, TimeLLM, TimesNet, iTransformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'LSTM': LSTM,
            'GRU': GRU,
            'DLinear': DLinear,
            'TimesNet': TimesNet,
            'Informer': Informer,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimeLLM': TimeLLM,
        }
        if args.model == 'TimeVLM':
            from src.TimeVLM import model as TimeVLM
            self.model_dict['TimeVLM'] = TimeVLM

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        if args.is_training:
            self._log_model_parameters()

    def _log_model_parameters(self):
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Learnable model parameters: {learnable_params:,}')
        print(f'Total model parameters: {total_params:,}')

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
