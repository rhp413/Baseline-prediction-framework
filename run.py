import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
from utils.tools import load_content


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def build_setting(args):
    data_stem = Path(args.data_path).stem
    suffix = f"_{args.setting_suffix}" if args.setting_suffix else ""
    return (
        f"{args.model}"
        f"__{data_stem}"
        f"__ft{args.features}"
        f"__sl{args.seq_len}"
        f"__ll{args.label_len}"
        f"__pl{args.pred_len}"
        f"__dm{args.d_model}"
        f"__bs{args.batch_size}"
        f"{suffix}"
    )


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Wind/PV long-term forecasting runner')

    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test')
    parser.add_argument('--model', type=str, default='Informer')

    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset')
    parser.add_argument('--data_path', type=str, default='wind.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='power')
    parser.add_argument('--freq', type=str, default='t')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--seq_len', type=int, default=288)
    parser.add_argument('--label_len', type=int, default=144)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=11)
    parser.add_argument('--dec_in', type=int, default=11)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=48)

    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    parser.add_argument('--vlm_type', type=str, default='CLIP')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--memory_bank_size', type=int, default=20)
    parser.add_argument('--patch_memory_size', type=int, default=100)
    parser.add_argument('--periodicity', type=int, default=96)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--three_channel_image', type=str2bool, default=True)
    parser.add_argument('--finetune_vlm', type=str2bool, default=False)
    parser.add_argument('--learnable_image', type=str2bool, default=True)
    parser.add_argument('--save_images', type=str2bool, default=False)
    parser.add_argument('--use_cross_attention', type=str2bool, default=True)
    parser.add_argument('--w_out_visual', type=str2bool, default=False)
    parser.add_argument('--w_out_text', type=str2bool, default=False)
    parser.add_argument('--w_out_query', type=str2bool, default=False)
    parser.add_argument('--visualize_embeddings', type=str2bool, default=False)
    parser.add_argument('--use_mem_gate', type=str2bool, default=False)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding', type=int, default=8)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--llm_layers', type=int, default=1)
    parser.add_argument('--prompt_domain', type=int, default=0)
    parser.add_argument('--align_const', type=float, default=0.4)
    parser.add_argument('--wo_ts', type=int, default=0)

    parser.add_argument('--percent', type=float, default=1)
    parser.add_argument('--result_root', type=str, default='./benchmark_results')
    parser.add_argument('--setting_suffix', type=str, default='')
    parser.add_argument('--save_txt', type=str2bool, default=True)
    parser.add_argument('--test_input_noise', action='store_true')
    parser.add_argument('--test_noise_level', type=float, default=0.10)
    parser.add_argument('--test_noise_seed', type=int, default=2024)
    parser.add_argument('--test_noise_include_target', action='store_true')

    args = parser.parse_args()
    if args.task_name != 'long_term_forecast':
        raise ValueError('This cleaned runner only supports long_term_forecast.')

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(x) for x in device_ids]
        args.gpu = args.device_ids[0]

    args.content = load_content(args)
    print_args(args)

    Exp = Exp_Long_Term_Forecast
    if args.is_training:
        for _ in range(args.itr):
            exp = Exp(args)
            setting = build_setting(args)
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            metrics = exp.test(setting)
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        setting = build_setting(args)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        metrics = exp.test(setting, test=1)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        torch.cuda.empty_cache()
