if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=336
seq_len=32
model_name=PHEnergyPatchMulti

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
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
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.3\
      --pha_hidden_H 128\
      --pha_lambda_energy 0.01 \
      --pha_lambda_struct 0.1 \
      --pha_lambda_moment 0.005 \
      --pha_lambda_diff_reg 0.1 \
      --pha_dt 0.05\
      --train_epochs 5\
      --patience 10\
      --itr 1\
      --batch_size 4 \
      --pha_mem_debug 0\
      --pha_use_hamiltonian 1\
      --pha_use_attention 1\
      --pha_use_diffusion 1\
      --pha_use_covariate 1\
      --pha_energy_mode H_diff\
      --learning_rate 3e-4 \
      --pha_loss_weighting uncertainty\
      --pha_bound_output 0\
      --use_amp >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done