if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=32
label_len=32
model_name=PHEnergyPatchMulti

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2021
# for pred_len in 96 192 336 720
for pred_len in 64
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
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --pha_hidden_H 128\
      --pha_lambda_energy 0.1 \
      --pha_lambda_struct 200 \
      --pha_lambda_moment 0.02 \
      --pha_lambda_diff_reg 2 \
      --pha_dt 0.05\
      --des 'Exp' \
      --train_epochs 1\
      --patience 10\
      --itr 1\
      --batch_size 4 \
      --pha_mem_debug 1\
      --pha_use_hamiltonian 1\
      --pha_use_attention 1\
      --pha_use_diffusion 1\
      --pha_use_covariate 1\
      --pha_energy_mode H_diff\
      --learning_rate 4e-4 \
      --pct_start 0.1\
      --pha_loss_weighting ema_norm\
      --pha_loss_ema_beta 0.99\
      --use_amp >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
