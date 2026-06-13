from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PHEnergyPatch, PHEnergyPatchMulti
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    @staticmethod
    def _unwrap_outputs(outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _get_model_obj(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _build_model(self):
        model_dict = {
            'PHEnergyPatch': PHEnergyPatch,
            'PHEnergyPatchMulti': PHEnergyPatchMulti,
        }
        model = model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_y_full = batch_y
                batch_x = batch_x.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs_raw = outputs
                outputs = self._unwrap_outputs(outputs)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                model_obj = self._get_model_obj()
                if hasattr(model_obj, "compute_loss"):
                    aux = outputs_raw[1] if isinstance(outputs_raw, tuple) else None
                    outputs_for_loss = (outputs, aux) if aux is not None else outputs
                    loss = model_obj.compute_loss(
                        outputs_for_loss,
                        batch_y,
                        batch_x,
                        batch_y_full.to(self.device),
                        batch_x_mark,
                        batch_y_mark,
                    )
                else:
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_y_full = batch_y
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device)

                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs_raw = outputs
                        outputs = self._unwrap_outputs(outputs)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        model_obj = self._get_model_obj()
                        if hasattr(model_obj, "compute_loss"):
                            aux = outputs_raw[1] if isinstance(outputs_raw, tuple) else None
                            outputs_for_loss = (outputs, aux) if aux is not None else outputs
                            loss = model_obj.compute_loss(
                                outputs_for_loss,
                                batch_y,
                                batch_x,
                                batch_y_full.to(self.device),
                                batch_x_mark,
                                batch_y_mark,
                            )
                        else:
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    outputs_raw = outputs
                    outputs = self._unwrap_outputs(outputs)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    model_obj = self._get_model_obj()
                    if hasattr(model_obj, "compute_loss"):
                        aux = outputs_raw[1] if isinstance(outputs_raw, tuple) else None
                        outputs_for_loss = (outputs, aux) if aux is not None else outputs
                        loss = model_obj.compute_loss(
                            outputs_for_loss,
                            batch_y,
                            batch_x,
                            batch_y_full.to(self.device),
                            batch_x_mark,
                            batch_y_mark,
                        )
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        def _pad_marks(mark_seq, pad_len):
            if pad_len <= 0:
                return mark_seq
            if mark_seq.numel() == 0:
                pad_val = torch.zeros(
                    mark_seq.size(0),
                    pad_len,
                    mark_seq.size(-1),
                    device=mark_seq.device,
                    dtype=mark_seq.dtype,
                )
                return pad_val
            last = mark_seq[:, -1:, :].repeat(1, pad_len, 1)
            return torch.cat([mark_seq, last], dim=1)

        def _free_run_autoregressive(batch_x, batch_x_mark, batch_y_mark):
            cur = batch_x
            cur_mark = batch_x_mark
            preds = []
            use_direct = self.args.model in ["Linear", "NLinear", "DLinear", "PatchTST"]

            for step in range(self.args.pred_len):
                if use_direct:
                    step_out = self.model(cur)
                else:
                    dec_hist = cur[:, -self.args.label_len:, :]
                    dec_zeros = torch.zeros(
                        cur.size(0),
                        self.args.pred_len,
                        cur.size(2),
                        device=cur.device,
                        dtype=cur.dtype,
                    )
                    dec_inp = torch.cat([dec_hist, dec_zeros], dim=1)

                    dec_mark_hist = cur_mark[:, -self.args.label_len:, :]
                    future_mark = batch_y_mark[
                        :,
                        self.args.label_len + step : self.args.label_len + step + self.args.pred_len,
                        :,
                    ]
                    future_mark = _pad_marks(future_mark, self.args.pred_len - future_mark.size(1))
                    dec_mark = torch.cat([dec_mark_hist, future_mark], dim=1)

                    if self.args.output_attention:
                        step_out = self.model(cur, cur_mark, dec_inp, dec_mark)[0]
                    else:
                        step_out = self.model(cur, cur_mark, dec_inp, dec_mark)

                step_out = self._unwrap_outputs(step_out)
                step_pred = step_out[:, :1, :]
                preds.append(step_pred)

                next_mark = batch_y_mark[
                    :,
                    self.args.label_len + step : self.args.label_len + step + 1,
                    :,
                ]
                if next_mark.size(1) == 0:
                    next_mark = cur_mark[:, -1:, :]

                cur = torch.cat([cur[:, 1:, :], step_pred], dim=1)
                cur_mark = torch.cat([cur_mark[:, 1:, :], next_mark], dim=1)

            return torch.cat(preds, dim=1)

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + self.args.data_path + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        def _long_free_run_full(test_data):
            data_x = torch.from_numpy(test_data.data_x).to(self.device)
            data_mark = torch.from_numpy(test_data.data_stamp).to(self.device)
            total_steps = data_x.size(0) - self.args.seq_len
            if total_steps <= 0:
                raise ValueError("Not enough test data for long free-run evaluation.")

            cur = data_x[: self.args.seq_len].unsqueeze(0)
            cur_mark = data_mark[: self.args.seq_len].unsqueeze(0)
            preds = []
            use_direct = self.args.model in ["Linear", "NLinear", "DLinear", "PatchTST"]

            for step in range(total_steps):
                if use_direct:
                    step_out = self.model(cur)
                else:
                    dec_hist = cur[:, -self.args.label_len:, :]
                    dec_zeros = torch.zeros(
                        cur.size(0),
                        self.args.pred_len,
                        cur.size(2),
                        device=cur.device,
                        dtype=cur.dtype,
                    )
                    dec_inp = torch.cat([dec_hist, dec_zeros], dim=1)

                    dec_mark_hist = cur_mark[:, -self.args.label_len:, :]
                    start = self.args.seq_len + step
                    future_mark = data_mark[start : start + self.args.pred_len].unsqueeze(0)
                    future_mark = _pad_marks(future_mark, self.args.pred_len - future_mark.size(1))
                    dec_mark = torch.cat([dec_mark_hist, future_mark], dim=1)

                    if self.args.output_attention:
                        step_out = self.model(cur, cur_mark, dec_inp, dec_mark)[0]
                    else:
                        step_out = self.model(cur, cur_mark, dec_inp, dec_mark)

                step_out = self._unwrap_outputs(step_out)
                step_pred = step_out[:, :1, :]
                preds.append(step_pred)

                next_mark = data_mark[
                    self.args.seq_len + step : self.args.seq_len + step + 1
                ].unsqueeze(0)
                if next_mark.size(1) == 0:
                    next_mark = cur_mark[:, -1:, :]

                cur = torch.cat([cur[:, 1:, :], step_pred], dim=1)
                cur_mark = torch.cat([cur_mark[:, 1:, :], next_mark], dim=1)

            preds = torch.cat(preds, dim=1)
            trues = data_x[self.args.seq_len : self.args.seq_len + total_steps].unsqueeze(0)
            inputx = data_x[: self.args.seq_len].unsqueeze(0)
            return preds, trues, inputx

        self.model.eval()
        with torch.no_grad():
            if self.args.long_free_run:
                outputs, batch_y, batch_x = _long_free_run_full(test_data)
                outputs = self._unwrap_outputs(outputs)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, 'long_free_run.pdf'))
            else:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    batch_x_mark = batch_x_mark.to(self.device)
                    batch_y_mark = batch_y_mark.to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
                    # encoder - decoder (standard test mode)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if 'Linear' in self.args.model or 'TST' in self.args.model:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = self._unwrap_outputs(outputs)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    # print(outputs.shape,batch_y.shape)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                    true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.detach().cpu().numpy())
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)

                # decoder input
                dec_inp = torch.zeros(
                    [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]],
                    device=batch_y.device,
                    dtype=batch_y.dtype,
                )
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = self._unwrap_outputs(outputs)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
