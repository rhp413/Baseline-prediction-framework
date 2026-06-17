import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric_extended
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.params_k = None
        self.max_memory_mb = None
        self.avg_train_time_s_per_epoch = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.params_k = total_params / 1000.0
        print(f"Params (K)      : {self.params_k:.2f} K")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _ensure_dir(self, path):
        os.makedirs(path, exist_ok=True)

    def _result_dir(self, setting):
        result_root = getattr(self.args, "result_root", "./benchmark_results")
        result_dir = os.path.join(result_root, setting)
        self._ensure_dir(result_dir)
        return result_dir

    def _inverse_and_select_target(self, outputs, batch_y, dataset_obj):
        """
        outputs, batch_y:
            numpy, shape [B, pred_len, C]
        对 inverse=True 的 MS 任务做逆标准化，并取目标列。
        """
        f_dim = -1 if self.args.features == 'MS' else 0
        n_features = len(dataset_obj.scaler.scale_) if getattr(dataset_obj, "scale", False) else outputs.shape[-1]

        if getattr(dataset_obj, "scale", False) and self.args.inverse:
            out_shape = outputs.shape
            y_shape = batch_y.shape

            out_2d = outputs.reshape(out_shape[0] * out_shape[1], out_shape[2])
            y_2d = batch_y.reshape(y_shape[0] * y_shape[1], y_shape[2])

            if out_2d.shape[1] != n_features:
                out_pad = np.zeros((out_2d.shape[0], n_features), dtype=out_2d.dtype)
                out_pad[:, -out_2d.shape[1]:] = out_2d
                out_2d = dataset_obj.inverse_transform(out_pad)[:, -out_shape[2]:]
            else:
                out_2d = dataset_obj.inverse_transform(out_2d)

            if y_2d.shape[1] != n_features:
                y_pad = np.zeros((y_2d.shape[0], n_features), dtype=y_2d.dtype)
                y_pad[:, -y_2d.shape[1]:] = y_2d
                y_2d = dataset_obj.inverse_transform(y_pad)[:, -y_shape[2]:]
            else:
                y_2d = dataset_obj.inverse_transform(y_2d)

            outputs = out_2d.reshape(out_shape)
            batch_y = y_2d.reshape(y_shape)

        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]
        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = float(np.average(total_loss)) if len(total_loss) > 0 else 0.0
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        self._ensure_dir(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        if self.args.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        epoch_times = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        target = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, target)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    target = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, target)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            one_epoch_time = time.time() - epoch_time
            epoch_times.append(one_epoch_time)
            print(f"Epoch: {epoch + 1} cost time: {one_epoch_time:.4f}")

            train_loss = float(np.average(train_loss)) if len(train_loss) > 0 else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        if self.args.use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Mem (MB)        : {self.max_memory_mb:.2f} MB")
        else:
            self.max_memory_mb = None
            print("Mem (MB)        : CPU mode")

        self.avg_train_time_s_per_epoch = float(np.mean(epoch_times)) if len(epoch_times) > 0 else None
        if self.avg_train_time_s_per_epoch is not None:
            print(f"ttrain (s/ep)   : {self.avg_train_time_s_per_epoch:.4f}")

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

        result_dir = self._result_dir(setting)
        self._ensure_dir(result_dir)

        preds = []
        trues = []
        total_infer_time = 0.0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y_device = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_y_for_dec = batch_y_device.clone()

                if getattr(self.args, 'test_input_noise', False):
                    noise_level = getattr(self.args, 'test_noise_level', 0.10)
                    noise_seed = getattr(self.args, 'test_noise_seed', 2024)
                    include_target = getattr(self.args, 'test_noise_include_target', False)

                    # 不要每个 batch 都重置成同一个 seed，用 seed + batch index 保证可复现且不同 batch 噪声不同
                    gen = torch.Generator(device=batch_x.device)
                    gen.manual_seed(noise_seed + i)

                    # batch_x: [B, seq_len, C]
                    channel_std_x = torch.std(batch_x, dim=(0, 1), unbiased=False)
                    noise_x = torch.randn(
                        batch_x.shape,
                        device=batch_x.device,
                        dtype=batch_x.dtype,
                        generator=gen
                    ) * (noise_level * channel_std_x.view(1, 1, -1))

                    if not include_target:
                        noise_x[:, :, -1] = 0.0

                    # 注意：这里不要 clamp。batch_x 是归一化空间，不能按原始物理量的非负约束裁剪。
                    batch_x = batch_x + noise_x

                    # decoder input 的已知历史段也属于测试输入，应该同步加噪；
                    # 但 future label 部分仍保持 clean，只用于评价。
                    dec_hist = batch_y_for_dec[:, :self.args.label_len, :]
                    channel_std_dec = torch.std(dec_hist, dim=(0, 1), unbiased=False)
                    noise_dec = torch.randn(
                        dec_hist.shape,
                        device=dec_hist.device,
                        dtype=dec_hist.dtype,
                        generator=gen
                    ) * (noise_level * channel_std_dec.view(1, 1, -1))

                    if not include_target:
                        noise_dec[:, :, -1] = 0.0

                    batch_y_for_dec[:, :self.args.label_len, :] = dec_hist + noise_dec

                dec_inp = torch.zeros_like(batch_y_for_dec[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y_for_dec[:, :self.args.label_len, :], dec_inp],
                    dim=1
                ).float().to(self.device)

                if self.args.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_start = time.perf_counter()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_gpu and torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_end = time.perf_counter()

                total_infer_time += (infer_end - infer_start)
                total_samples += batch_x.shape[0]

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_np = batch_y_device[:, -self.args.pred_len:, :].detach().cpu().numpy()
                outputs_np = outputs.detach().cpu().numpy()

                outputs_np, batch_y_np = self._inverse_and_select_target(outputs_np, batch_y_np, test_data)

                pred = outputs_np
                true = batch_y_np

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if getattr(test_data, "scale", False) and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    try:
                        gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(result_dir, f'vis_{i}.pdf'))
                    except Exception:
                        pass

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)

        dtw_val = None

        mae, sse, mse, rmse, mape, mspe, r2 = metric_extended(preds, trues)
        tinf_ms = (total_infer_time / total_samples) * 1000.0 if total_samples > 0 else None

        metrics_dict = {
            'setting': setting,
            'model': self.args.model,
            'data_path': self.args.data_path,
            'target': self.args.target,
            'features': self.args.features,
            'seq_len': int(self.args.seq_len),
            'label_len': int(self.args.label_len),
            'pred_len': int(self.args.pred_len),
            'mae': float(mae),
            'sse': float(sse),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'r2': float(r2),
            'params_k': float(self.params_k) if self.params_k is not None else None,
            'mem_mb': float(self.max_memory_mb) if self.max_memory_mb is not None else None,
            't_train_s_per_ep': float(self.avg_train_time_s_per_epoch) if self.avg_train_time_s_per_epoch is not None else None,
            't_inf_ms': float(tinf_ms) if tinf_ms is not None else None,
            'dtw': float(dtw_val) if dtw_val is not None else None,
        }

        print(json.dumps(metrics_dict, indent=2, ensure_ascii=False))

        np.save(os.path.join(result_dir, 'pred.npy'), preds)
        np.save(os.path.join(result_dir, 'true.npy'), trues)
        np.save(
            os.path.join(result_dir, 'metrics.npy'),
            np.array([mae, sse, mse, rmse, r2], dtype=np.float64)
        )

        with open(os.path.join(result_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

        if getattr(self.args, "save_txt", True):
            report_path = os.path.join(result_dir, 'run_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"setting        : {setting}\n")
                f.write(f"model          : {self.args.model}\n")
                f.write(f"dataset        : {self.args.data_path}\n")
                f.write(f"target         : {self.args.target}\n")
                f.write(f"seq_len        : {self.args.seq_len}\n")
                f.write(f"label_len      : {self.args.label_len}\n")
                f.write(f"pred_len       : {self.args.pred_len}\n")
                f.write(f"MAE            : {mae:.10f}\n")
                f.write(f"SSE            : {sse:.10f}\n")
                f.write(f"MSE            : {mse:.10f}\n")
                f.write(f"RMSE           : {rmse:.10f}\n")
                f.write(f"R2             : {r2:.10f}\n")
                f.write(f"Params (K)     : {self.params_k if self.params_k is not None else 'NA'}\n")
                f.write(f"Mem (MB)       : {self.max_memory_mb if self.max_memory_mb is not None else 'NA'}\n")
                f.write(f"ttrain (s/ep)  : {self.avg_train_time_s_per_epoch if self.avg_train_time_s_per_epoch is not None else 'NA'}\n")
                f.write(f"tinf (ms)      : {tinf_ms if tinf_ms is not None else 'NA'}\n")
                f.write(f"DTW            : {dtw_val if dtw_val is not None else 'not calculated'}\n")

        return metrics_dict
