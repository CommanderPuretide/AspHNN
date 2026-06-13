if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=336
seq_len=48
model_name=PHEnergyPatch

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021
# for pred_len in 96 192 336 720
for pred_len in 1
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --lradj 'TST'\
      --pha_hidden_H 128\
      --pha_lambda_energy 0.01 \
      --pha_lambda_struct 1 \
      --pha_lambda_moment 0.005 \
      --pha_lambda_diff_reg 0.2 \
      --pha_dt 0.05\
      --des 'Exp' \
      --train_epochs 20\
      --patience 10\
      --itr 1\
      --batch_size 128 \
      --pha_mem_debug 0\
      --pha_use_hamiltonian 1\
      --pha_use_attention 1\
      --pha_use_diffusion 1\
      --pha_use_covariate 1\
      --pha_energy_mode H_diff\
      --learning_rate 1e-4 \
      --pct_start 0.4\
      --pha_loss_weighting uncertainty\
      --pha_midpoint_iters 1\
      --pha_bound_output 0\
      --pha_init_u_scale 1\
      --use_amp >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done


    #   --pha_loss_weighting ema_norm\
    #   --pha_loss_ema_beta 0.99\
    #   --pha_loss_weight_min 1e-3\
    #   --pha_loss_weight_max 2.0\    


    #   --pha_lambda_energy 0.01 \
    #   --pha_lambda_struct 1 \
    #   --pha_lambda_moment 0.005 \
    #   --pha_lambda_diff_reg 0.2 \