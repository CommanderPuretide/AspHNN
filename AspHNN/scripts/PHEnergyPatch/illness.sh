if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=104
seq_len=48
model_name=PHEnergyPatch

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021
# for pred_len in 24 36 48 60
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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 50\
      --patience 10\
      --itr 1\
      --batch_size 8 \
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
      --pha_midpoint_iters 1\
      --use_amp >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done