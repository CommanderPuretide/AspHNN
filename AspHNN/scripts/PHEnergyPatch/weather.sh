export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=32
model_name=PHEnergyPatch

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
# for pred_len in 96 192 336 720
for pred_len in 8
# do
#     python -u run_longExp.py \
#       --random_seed $random_seed \
#       --is_training 1 \
#       --root_path $root_path_name \
#       --data_path $data_path_name \
#       --model_id $model_id_name_$seq_len'_'$pred_len \
#       --model $model_name \
#       --data $data_name \
#       --features M \
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 21 \
#       --e_layers 3 \
#       --n_heads 16 \
#       --d_model 128 \
#       --d_ff 256 \
#       --dropout 0.2\
#       --fc_dropout 0.2\
#       --head_dropout 0\
#       --patch_len 16\
#       --stride 8\
#       --pha_hidden_H 128\
#       --pha_lambda_energy 0.05\
#       --pha_lambda_struct 1e-3\
#       --pha_lambda_moment 0.05\
#       --pha_lambda_diff_reg 1e-5\
#       --pha_dt 0.2\
#       --des 'Exp' \
#       --train_epochs 20\
#       --patience 20\
#       --itr 1\
#       --batch_size 128 \
#       --learning_rate 0.0001 \
#       --use_amp\
#       --pha_mem_debug 1\
#       --pha_use_hamiltonian 1\
#       --pha_use_attention 1\
#       --pha_use_diffusion 1\
#       --pha_use_covariate 0\
#       --pha_energy_mode H_diff\
#       --long_free_run\
#       --pha_midpoint_iters 1 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
# done
do
    python -u run_longExp.py \
        --random_seed 2021 \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path weather.csv \
        --model_id weather_${seq_len}_${pred_len} \
        --model PHEnergyPatch \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 21 \
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
        --train_epochs 10\
        --patience 1\
        --pha_hidden_H 128 \
        --pha_lambda_energy 0.1 \
        --pha_lambda_struct 0.1 \
        --pha_lambda_moment 0.1 \
        --pha_lambda_diff_reg 0.1 \
        --pha_dt 0.01 \
        --pha_use_hamiltonian 1 \
        --pha_use_attention 1 \
        --pha_use_diffusion 1 \
        --pha_diff_scale 0.5 \
        --pha_use_covariate 1 \
        --pha_energy_mode H_diff \
        --pha_midpoint_iters 1 \
        --pha_bound_output 0 \
        --pha_init_u_scale 1.0 \
        --itr 1 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --pha_loss_weighting uncertainty\
        --pha_lambda_attn 0.001 \
        --pha_attn_reg locality\
        --use_amp > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
