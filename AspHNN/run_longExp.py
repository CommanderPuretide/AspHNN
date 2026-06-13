import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='PHEnergyPatch',
                        help='model name, options: [PHEnergyPatch, PHEnergyPatchMulti]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--long_free_run', action='store_true',
                        help='run a single long free-run over the full test segment')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # PHEnergyPatch
    parser.add_argument('--pha_hidden_H', type=int, default=96, help='hidden size for Hamiltonian MLP')
    parser.add_argument('--pha_use_covariate', type=int, default=0, help='use time covariates; True 1 False 0')
    parser.add_argument('--pha_mem_debug', type=int, default=0, help='print cuda memory summary once; True 1 False 0')
    parser.add_argument('--pha_midpoint_iters', type=int, default=4, help='midpoint iterations for PH rollout')
    parser.add_argument('--pha_dt', type=float, default=0.7, help='time step for PH rollout')
    parser.add_argument('--pha_use_hamiltonian', type=int, default=1, help='use Hamiltonian term; True 1 False 0')
    parser.add_argument('--pha_use_attention', type=int, default=1, help='use attention residual; True 1 False 0')
    parser.add_argument('--pha_use_diffusion', type=int, default=1, help='use diffusion term; True 1 False 0')
    parser.add_argument('--pha_lambda_energy', type=float, default=0.05, help='energy regularization weight')
    parser.add_argument('--pha_lambda_struct', type=float, default=1e-4, help='structure regularization weight')
    parser.add_argument('--pha_lambda_moment', type=float, default=1.0, help='moment matching weight')
    parser.add_argument('--pha_lambda_diff_reg', type=float, default=1e-4, help='diffusion regularization weight')
    parser.add_argument('--pha_lambda_attn', type=float, default=0.0, help='attention regularization weight')
    parser.add_argument(
        '--pha_attn_reg',
        type=str,
        default='locality',
        choices=['locality', 'l2', 'entropy'],
        help='attention regularizer type',
    )
    parser.add_argument(
        '--pha_loss_weighting',
        type=str,
        default='fixed',
        choices=['fixed', 'uncertainty', 'ema_norm'],
        help='loss weighting mode for multi-term loss',
    )
    parser.add_argument(
        '--pha_loss_ema_beta',
        type=float,
        default=0.99,
        help='EMA decay for ema_norm loss weighting',
    )
    parser.add_argument(
        '--pha_loss_ema_eps',
        type=float,
        default=1e-6,
        help='epsilon for ema_norm loss weighting',
    )
    parser.add_argument(
        '--pha_loss_weight_min',
        type=float,
        default=1e-3,
        help='min clamp for dynamic loss weights',
    )
    parser.add_argument(
        '--pha_loss_weight_max',
        type=float,
        default=10.0,
        help='max clamp for dynamic loss weights',
    )
    parser.add_argument('--pha_cov_rollout', type=str, default='last', help="covariate rollout: 'last' or 'zero'")
    parser.add_argument('--pha_energy_mode', type=str, default='H_diff', help="energy mode: 'H_diff' or 'grad_diff'")
    parser.add_argument('--pha_diff_rank', type=int, default=4, help='diffusion factor rank')
    parser.add_argument('--pha_diff_hidden', type=int, default=64, help='diffusion MLP hidden size')
    parser.add_argument('--pha_diff_scale', type=float, default=1.0, help='diffusion scale')
    parser.add_argument('--pha_midpoint_tol', type=float, default=0.0, help='midpoint convergence tolerance')
    parser.add_argument('--pha_pt_d_model', type=int, default=128, help='PatchTST d_model override')
    parser.add_argument('--pha_pt_nlayers', type=int, default=2, help='PatchTST e_layers override')
    parser.add_argument('--pha_pt_nhead', type=int, default=8, help='PatchTST n_heads override')
    parser.add_argument('--pha_pt_d_ff', type=int, default=256, help='PatchTST d_ff override')
    parser.add_argument('--pha_pt_dropout', type=float, default=0.05, help='PatchTST dropout override')
    parser.add_argument('--pha_pt_fc_dropout', type=float, default=0.05, help='PatchTST fc_dropout override')
    parser.add_argument('--pha_pt_head_dropout', type=float, default=0.0, help='PatchTST head_dropout override')
    parser.add_argument('--pha_pt_patch_len', type=int, default=16, help='PatchTST patch_len override')
    parser.add_argument('--pha_pt_stride', type=int, default=8, help='PatchTST stride override')
    parser.add_argument('--pha_pt_padding_patch', type=str, default='end', help='PatchTST padding_patch override')
    parser.add_argument('--pha_pt_attn_dropout', type=float, default=0.0, help='PatchTST attn_dropout override')
    parser.add_argument('--pha_pt_norm', type=str, default='BatchNorm', help='PatchTST norm override')
    parser.add_argument('--pha_pt_activation', type=str, default='gelu', help='PatchTST activation override')
    parser.add_argument('--pha_pt_key_padding_mask', type=str, default='auto', help='PatchTST key_padding_mask override')
    parser.add_argument('--pha_pt_pre_norm', type=int, default=0, help='PatchTST pre_norm; True 1 False 0')
    parser.add_argument('--pha_pt_store_attn', type=int, default=0, help='PatchTST store_attn; True 1 False 0')
    parser.add_argument('--pha_pt_pe', type=str, default='zeros', help='PatchTST positional encoding type')
    parser.add_argument('--pha_pt_learn_pe', type=int, default=1, help='PatchTST learn_pe; True 1 False 0')
    parser.add_argument('--pha_pt_revin', type=int, default=1, help='PatchTST revin override; True 1 False 0')
    parser.add_argument('--pha_pt_affine', type=int, default=0, help='PatchTST affine override; True 1 False 0')
    parser.add_argument('--pha_pt_subtract_last', type=int, default=0, help='PatchTST subtract_last override; True 1 False 0')
    parser.add_argument('--pha_bound_output', type=int, default=1, help='bound output with tanh; True 1 False 0')
    parser.add_argument('--pha_init_u_scale', type=float, default=1.0, help='initial output scaling')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.label_len > args.seq_len:
        print(
            f"[warning] label_len ({args.label_len}) > seq_len ({args.seq_len}); "
            "clamping label_len to seq_len."
        )
        args.label_len = args.seq_len

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        
