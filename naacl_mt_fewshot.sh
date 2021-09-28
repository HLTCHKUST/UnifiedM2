
echo "=================================train few shot classifier based on vanilla RoBERTa================================="
export SAVE_PATH=/home/nayeon/misinfo/save # path to save model, checkpoint, etc
export DATA_PATH=/home/nayeon/misinfo/data  #path to processed data

export BSZ=1
export steps=1
export CUDA=4
FEWSHOT=0.001
SEED=42
LR=5e-6 



export TASK_NAME=fnn_politifact # options for TASK_NAME: webis, clickbait, rumour_veracity, rumour_veracity_binary, basil_detection
CUDA_VISIBLE_DEVICES=$CUDA python main.py \
    --model_type roberta \
    --model_name_or_path 'roberta-large' \
    --config_name 'roberta-large' \
    --tokenizer_name 'roberta-large' \
    --task_name $TASK_NAME \
    --do_train \
    --do_test \
    --do_lower_case \
    --data_dir $DATA_PATH \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BSZ   \
    --per_gpu_train_batch_size=$BSZ   \
    --learning_rate $LR \
    --patience 5 \
    --num_train_epochs 5.0 \
    --output_dir $SAVE_PATH/$TASK_NAME/ \
    --main_task_w 1 \
    --aux_task_w 0.1 \
    --evaluate_during_training \
    --root_dir $DATA_PATH \
    --gradient_accumulation_steps $steps \
    --overwrite_output_dir \
    --fewshot_train_ratio $FEWSHOT \
    --seed $SEED \
    --custom_exp_name vanilla$FEWSHOT.s$SEED




export SAVE_PATH=/home/nayeon/misinfo/save # path to save model, checkpoint, etc
export DATA_PATH=/home/nayeon/misinfo/data  #path to processed data

export BSZ=1
export steps=1
export CUDA=6
FEWSHOT=0.1
SEED=42
LR=5e-6 



export TASK_NAME=fnn_buzzfeed # options for TASK_NAME: webis, clickbait, rumour_veracity, rumour_veracity_binary, basil_detection
CUDA_VISIBLE_DEVICES=$CUDA python main.py \
    --model_type roberta \
    --model_name_or_path 'roberta-large' \
    --config_name 'roberta-large' \
    --tokenizer_name 'roberta-large' \
    --task_name $TASK_NAME \
    --do_train \
    --do_test \
    --do_lower_case \
    --data_dir $DATA_PATH \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BSZ   \
    --per_gpu_train_batch_size=$BSZ   \
    --learning_rate $LR \
    --patience 5 \
    --num_train_epochs 5.0 \
    --output_dir $SAVE_PATH/$TASK_NAME/ \
    --main_task_w 1 \
    --aux_task_w 0.1 \
    --evaluate_during_training \
    --root_dir $DATA_PATH \
    --gradient_accumulation_steps $steps \
    --overwrite_output_dir \
    --fewshot_train_ratio $FEWSHOT \
    --seed $SEED \
    --custom_exp_name vanilla.$FEWSHOT.s$SEED
















export BSZ=8
export steps=4
export CUDA=5
FEWSHOT=-1
SEED=42
LR=5e-6

# export TASK_NAME=fnn_buzzfeed
# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
#     --model_type roberta \
#     --model_name_or_path 'roberta-large' \
#     --config_name 'roberta-large' \
#     --tokenizer_name 'roberta-large' \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_test \
#     --do_lower_case \
#     --data_dir $DATA_PATH \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BSZ   \
#     --per_gpu_train_batch_size=$BSZ   \
#     --learning_rate $LR \
#     --patience 5 \
#     --num_train_epochs 5.0 \
#     --output_dir $SAVE_PATH/$TASK_NAME/ \
#     --main_task_w 1 \
#     --aux_task_w 0.1 \
#     --evaluate_during_training \
#     --root_dir $DATA_PATH \
#     --gradient_accumulation_steps $steps \
#     --overwrite_output_dir \
#     --fewshot_train $FEWSHOT \
#     --seed $SEED \
#     --custom_exp_name vanilla$FEWSHOT.s$SEED &


export TASK_NAME=fnn_politifact
CUDA_VISIBLE_DEVICES=$CUDA python main.py \
    --model_type roberta \
    --model_name_or_path 'roberta-large' \
    --config_name 'roberta-large' \
    --tokenizer_name 'roberta-large' \
    --task_name $TASK_NAME \
    --do_train \
    --do_test \
    --do_lower_case \
    --data_dir $DATA_PATH \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BSZ   \
    --per_gpu_train_batch_size=$BSZ   \
    --learning_rate $LR \
    --patience 5 \
    --num_train_epochs 5.0 \
    --output_dir $SAVE_PATH/$TASK_NAME/ \
    --main_task_w 1 \
    --aux_task_w 0.1 \
    --evaluate_during_training \
    --root_dir $DATA_PATH \
    --gradient_accumulation_steps $steps \
    --overwrite_output_dir \
    --fewshot_train $FEWSHOT \
    --seed $SEED \
    --custom_exp_name vanilla$FEWSHOT.s$SEED &

# export BSZ=1
# export steps=1
# export CUDA=2
# FEWSHOT=50
# # default seed

# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
#     --model_type roberta \
#     --model_name_or_path 'roberta-large' \
#     --config_name 'roberta-large' \
#     --tokenizer_name 'roberta-large' \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_test \
#     --do_lower_case \
#     --data_dir $DATA_PATH \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BSZ   \
#     --per_gpu_train_batch_size=$BSZ   \
#     --learning_rate $LR \
#     --patience 5 \
#     --num_train_epochs 5.0 \
#     --output_dir $SAVE_PATH/$TASK_NAME/ \
#     --main_task_w 1 \
#     --aux_task_w 0.1 \
#     --evaluate_during_training \
#     --root_dir $DATA_PATH \
#     --gradient_accumulation_steps $steps \
#     --overwrite_output_dir \
#     --fewshot_train $FEWSHOT \
#     --custom_exp_name vanilla$FEWSHOT &



# echo "fnn_buzzfeed"
# export BSZ=1
# export steps=1
# export CUDA=6
# FEWSHOT=0.2

# export TASK_NAME=fnn_buzzfeed_title
# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
#     --model_type roberta \
#     --model_name_or_path 'roberta-large' \
#     --config_name 'roberta-large' \
#     --tokenizer_name 'roberta-large' \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_test \
#     --do_lower_case \
#     --data_dir $DATA_PATH \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BSZ   \
#     --per_gpu_train_batch_size=$BSZ   \
#     --learning_rate 5e-6 \
#     --patience 5 \
#     --num_train_epochs 5.0 \
#     --output_dir $SAVE_PATH/$TASK_NAME/ \
#     --main_task_w 1 \
#     --aux_task_w 0.1 \
#     --evaluate_during_training \
#     --root_dir $DATA_PATH \
#     --gradient_accumulation_steps $steps \
#     --overwrite_output_dir \
#     --fewshot_train $FEWSHOT \
#     --custom_exp_name vanilla$FEWSHOT 



# echo "=================================train few shot classifier based on {ST} RoBERTa================================="
export SAVE_PATH=/home/nayeon/misinfo/save # path to save model, checkpoint, etc
export DATA_PATH=/home/nayeon/misinfo/data  #path to processed data
export BSZ=1
export steps=1


export TASK_NAME=fnn_politifact
FEWSHOT=0.1
ST_BASIL=/home/nayeon/misinfo/save/basil_detection/lr.5e-06_bz.8__roberta_exp1
ST_CLICKBAIT=/home/nayeon/misinfo/save/clickbait/lr.5e-06_bz.8__roberta_exp1
ST_RUMOUR=/home/nayeon/misinfo/save/rumour_veracity_binary/lr.5e-06_bz.8__roberta_exp1
ST_WEBIS=/home/nayeon/misinfo/save/webis/lr.5e-06_bz.8__roberta_exp1

ST_MODELS=($ST_BASIL $ST_CLICKBAIT $ST_RUMOUR $ST_WEBIS)
NAMES=(st_basil st_clickbait st_rumour st_webis)
CUDA=3

ID=3
CUDA_VISIBLE_DEVICES=$CUDA python main.py \
    --model_type roberta \
    --model_name_or_path ${ST_MODELS[$ID]}/best_model \
    --config_name 'roberta-large' \
    --tokenizer_name 'roberta-large' \
    --task_name $TASK_NAME \
    --do_train \
    --do_test \
    --do_lower_case \
    --data_dir "" \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BSZ   \
    --per_gpu_train_batch_size=$BSZ   \
    --learning_rate 5e-6 \
    --patience 5 \
    --num_train_epochs 5.0 \
    --output_dir $SAVE_PATH/$TASK_NAME/ \
    --main_task_w 1 \
    --aux_task_w 0.1 \
    --evaluate_during_training \
    --root_dir $DATA_PATH \
    --gradient_accumulation_steps $steps \
    --overwrite_output_dir \
    --fewshot_train_ratio $FEWSHOT \
    --custom_exp_name ${NAMES[$ID]}$FEWSHOT

# ID=1
# CUDA_VISIBLE_DEVICES=${CUDAS[$ID]} python main.py \
#     --model_type roberta \
#     --model_name_or_path ${ST_MODELS[$ID]}/best_model \
#     --config_name 'roberta-large' \
#     --tokenizer_name 'roberta-large' \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_test \
#     --do_lower_case \
#     --data_dir $DATA_PATH \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BSZ   \
#     --per_gpu_train_batch_size=$BSZ   \
#     --learning_rate 5e-6 \
#     --patience 5 \
#     --num_train_epochs 5.0 \
#     --output_dir $SAVE_PATH/$TASK_NAME/ \
#     --main_task_w 1 \
#     --aux_task_w 0.1 \
#     --evaluate_during_training \
#     --root_dir $DATA_PATH \
#     --gradient_accumulation_steps $steps \
#     --overwrite_output_dir \
#     --fewshot_train $FEWSHOT \
#     --custom_exp_name ${NAMES[$ID]}$FEWSHOT &

# ID=2
# CUDA_VISIBLE_DEVICES=${CUDAS[$ID]} python main.py \
#     --model_type roberta \
#     --model_name_or_path ${ST_MODELS[$ID]}/best_model \
#     --config_name 'roberta-large' \
#     --tokenizer_name 'roberta-large' \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_test \
#     --do_lower_case \
#     --data_dir $DATA_PATH \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BSZ   \
#     --per_gpu_train_batch_size=$BSZ   \
#     --learning_rate 5e-6 \
#     --patience 5 \
#     --num_train_epochs 5.0 \
#     --output_dir $SAVE_PATH/$TASK_NAME/ \
#     --main_task_w 1 \
#     --aux_task_w 0.1 \
#     --evaluate_during_training \
#     --root_dir $DATA_PATH \
#     --gradient_accumulation_steps $steps \
#     --overwrite_output_dir \
#     --fewshot_train $FEWSHOT \
#     --custom_exp_name ${NAMES[$ID]}$FEWSHOT &

# ID=3
# CUDA_VISIBLE_DEVICES=${CUDAS[$ID]} python main.py \
#     --model_type roberta \
#     --model_name_or_path ${ST_MODELS[$ID]}/best_model \
#     --config_name 'roberta-large' \
#     --tokenizer_name 'roberta-large' \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_test \
#     --do_lower_case \
#     --data_dir $DATA_PATH \
#     --max_seq_length 128 \
#     --per_gpu_eval_batch_size=$BSZ   \
#     --per_gpu_train_batch_size=$BSZ   \
#     --learning_rate 5e-6 \
#     --patience 5 \
#     --num_train_epochs 5.0 \
#     --output_dir $SAVE_PATH/$TASK_NAME/ \
#     --main_task_w 1 \
#     --aux_task_w 0.1 \
#     --evaluate_during_training \
#     --root_dir $DATA_PATH \
#     --gradient_accumulation_steps $steps \
#     --overwrite_output_dir \
#     --fewshot_train $FEWSHOT \
#     --custom_exp_name ${NAMES[$ID]}$FEWSHOT 


echo "========================few shot on FakeNewsNet Dataset {after MT}================================="

TASK_NAME=fnn_politifact
ROOT_DIR=/home/nayeon/misinfo/results
SAVE_DIR=$ROOT_DIR/$TASK_NAME/
DATA_DIR=/home/nayeon/misinfo/data 

MT_MODEL_PATH=/home/nayeon/misinfo/results/basil_detection,basil_type,basil_polarity,webis,clickbait,rumour_veracity/lr.5e-06_bz.1_dropout.0_decay.0.0_opt.adamw_stopRemove.False_use_ne_text.False_mtloss.1.0-0.1_bz.fit_main_bs.main_focus_lossf.xentropy_gamma.0.5_evalmetric_f1_mt_joint_loss_acc16
bsz=1
steps=1
LR=5e-6

FEWSHOT=0.1
SEED=42
CUDA=4

CUDA_VISIBLE_DEVICES=$CUDA python main.py \
--model_type roberta \
--model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
--task_name $TASK_NAME \
--do_train \
--do_test \
--do_lower_case \
--data_dir '' \
--tokenizer_name 'roberta-large' \
--max_seq_length 128 \
--per_gpu_eval_batch_size=$bsz   \
--per_gpu_train_batch_size=$bsz   \
--learning_rate $LR \
--num_train_epochs 10.0 \
--output_dir $SAVE_DIR \
--main_task_w 1 \
--aux_task_w 0.1 \
--patience 5 \
--evaluate_during_training \
--root_dir $DATA_DIR \
--b_scheduling main_focus \
--focal_gamma 0.5 \
--gradient_accumulation_steps $steps \
--fewshot_train_ratio $FEWSHOT \
--seed $SEED \
--custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT.s$SEED \
--overwrite_output_dir 




TASK_NAME=propaganda
ROOT_DIR=/home/nayeon/misinfo/results
SAVE_DIR=$ROOT_DIR/$TASK_NAME/
DATA_DIR=/home/nayeon/misinfo/data 

MT_MODEL_PATH=/home/nayeon/misinfo/results/basil_detection,basil_type,basil_polarity,webis,clickbait,rumour_veracity/lr.5e-06_bz.1_dropout.0_decay.0.0_opt.adamw_stopRemove.False_use_ne_text.False_mtloss.1.0-0.1_bz.fit_main_bs.main_focus_lossf.xentropy_gamma.0.5_evalmetric_f1_mt_joint_loss_acc16
bsz=8
steps=4
LR=5e-6

FEWSHOT=0.1
SEED=42
CUDA=3

CUDA_VISIBLE_DEVICES=$CUDA python main.py \
--model_type roberta \
--model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
--task_name $TASK_NAME \
--do_train \
--do_test \
--do_lower_case \
--data_dir '' \
--tokenizer_name 'roberta-large' \
--max_seq_length 128 \
--per_gpu_eval_batch_size=$bsz   \
--per_gpu_train_batch_size=$bsz   \
--learning_rate $LR \
--num_train_epochs 5.0 \
--output_dir $SAVE_DIR \
--main_task_w 1 \
--aux_task_w 0.1 \
--patience 5 \
--evaluate_during_training \
--root_dir $DATA_DIR \
--b_scheduling main_focus \
--focal_gamma 0.5 \
--gradient_accumulation_steps $steps \
--fewshot_train_ratio $FEWSHOT \
--seed $SEED \
--custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT.s$SEED \
--overwrite_output_dir 



# FEWSHOT=-1
# SEED=42
# CUDA=4
# TASK_NAME=fnn_buzzfeed
# LR=1e-5

# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
# --model_type roberta \
# --model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
# --task_name $TASK_NAME \
# --do_train \
# --do_test \
# --do_lower_case \
# --data_dir '' \
# --tokenizer_name 'roberta-large' \
# --max_seq_length 128 \
# --per_gpu_eval_batch_size=$bsz   \
# --per_gpu_train_batch_size=$bsz   \
# --learning_rate $LR \
# --num_train_epochs 5.0 \
# --output_dir $SAVE_DIR \
# --main_task_w 1 \
# --aux_task_w 0.1 \
# --patience 5 \
# --evaluate_during_training \
# --root_dir $DATA_DIR \
# --b_scheduling main_focus \
# --focal_gamma 0.5 \
# --gradient_accumulation_steps $steps \
# --fewshot_train $FEWSHOT \
# --seed $SEED \
# --custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT.s$SEED \
# --overwrite_output_dir &


# export TASK_NAME=fnn_politifact
# FEWSHOT=-1
# SEED=42
# CUDA=5
# LR=1e-5

# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
# --model_type roberta \
# --model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
# --task_name $TASK_NAME \
# --do_train \
# --do_test \
# --do_lower_case \
# --data_dir '' \
# --tokenizer_name 'roberta-large' \
# --max_seq_length 128 \
# --per_gpu_eval_batch_size=$bsz   \
# --per_gpu_train_batch_size=$bsz   \
# --learning_rate $LR \
# --num_train_epochs 5.0 \
# --output_dir $SAVE_DIR \
# --main_task_w 1 \
# --aux_task_w 0.1 \
# --patience 5 \
# --evaluate_during_training \
# --root_dir $DATA_DIR \
# --b_scheduling main_focus \
# --focal_gamma 0.5 \
# --gradient_accumulation_steps $steps \
# --fewshot_train $FEWSHOT \
# --seed $SEED \
# --custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT.s$SEED \
# --overwrite_output_dir &


# FEWSHOT=50
# # default seed
# CUDA=5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
# --model_type roberta \
# --model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
# --task_name $TASK_NAME \
# --do_train \
# --do_test \
# --do_lower_case \
# --data_dir '' \
# --tokenizer_name 'roberta-large' \
# --max_seq_length 128 \
# --per_gpu_eval_batch_size=$bsz   \
# --per_gpu_train_batch_size=$bsz   \
# --learning_rate $LR \
# --num_train_epochs 5.0 \
# --output_dir $SAVE_DIR \
# --main_task_w 1 \
# --aux_task_w 0.1 \
# --patience 5 \
# --evaluate_during_training \
# --root_dir $DATA_DIR \
# --b_scheduling main_focus \
# --focal_gamma 0.5 \
# --gradient_accumulation_steps $steps \
# --fewshot_train $FEWSHOT \
# --custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT \
# --overwrite_output_dir










# echo "fnn_politifact"
# TASK_NAME=fnn_politifact
# ROOT_DIR=/home/nayeon/misinfo/results
# SAVE_DIR=$ROOT_DIR/$TASK_NAME/
# DATA_DIR=/home/nayeon/misinfo/data 

# MT_MODEL_PATH=/home/nayeon/misinfo/results/basil_detection,basil_type,basil_polarity,webis,clickbait,rumour_veracity/lr.5e-06_bz.1_dropout.0_decay.0.0_opt.adamw_stopRemove.False_use_ne_text.False_mtloss.1.0-0.1_bz.fit_main_bs.main_focus_lossf.xentropy_gamma.0.5_evalmetric_f1_mt_joint_loss_acc16
# bsz=1
# steps=1 #4

# FEWSHOT=0.1
# CUDA=0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
# --model_type roberta \
# --model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
# --task_name $TASK_NAME \
# --do_train \
# --do_test \
# --do_lower_case \
# --data_dir '' \
# --tokenizer_name 'roberta-large' \
# --max_seq_length 128 \
# --per_gpu_eval_batch_size=$bsz   \
# --per_gpu_train_batch_size=$bsz   \
# --learning_rate 5e-6 \
# --num_train_epochs 5.0 \
# --output_dir $SAVE_DIR \
# --main_task_w 1 \
# --aux_task_w 0.1 \
# --patience 5 \
# --evaluate_during_training \
# --root_dir $DATA_DIR \
# --b_scheduling main_focus \
# --focal_gamma 0.5 \
# --gradient_accumulation_steps $steps \
# --fewshot_train $FEWSHOT \
# --custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT \
# --overwrite_output_dir


# echo "fnn_buzzfeed_title"
# TASK_NAME=fnn_buzzfeed_title
# ROOT_DIR=/home/nayeon/misinfo/results
# SAVE_DIR=$ROOT_DIR/$TASK_NAME/
# DATA_DIR=/home/nayeon/misinfo/data 

# MT_MODEL_PATH=/home/nayeon/misinfo/results/basil_detection,basil_type,basil_polarity,webis,clickbait,rumour_veracity/lr.5e-06_bz.1_dropout.0_decay.0.0_opt.adamw_stopRemove.False_use_ne_text.False_mtloss.1.0-0.1_bz.fit_main_bs.main_focus_lossf.xentropy_gamma.0.5_evalmetric_f1_mt_joint_loss_acc16
# bsz=1
# steps=1

# FEWSHOT=0.2
# CUDA=7
# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
# --model_type roberta \
# --model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
# --task_name $TASK_NAME \
# --do_train \
# --do_test \
# --do_lower_case \
# --data_dir '' \
# --tokenizer_name 'roberta-large' \
# --max_seq_length 128 \
# --per_gpu_eval_batch_size=$bsz   \
# --per_gpu_train_batch_size=$bsz   \
# --learning_rate 1e-6 \
# --num_train_epochs 5.0 \
# --output_dir $SAVE_DIR \
# --main_task_w 1 \
# --aux_task_w 0.1 \
# --patience 5 \
# --evaluate_during_training \
# --root_dir $DATA_DIR \
# --b_scheduling main_focus \
# --focal_gamma 0.5 \
# --gradient_accumulation_steps $steps \
# --fewshot_train $FEWSHOT \
# --custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT \
# --overwrite_output_dir


# TASK_NAME=fnn_buzzfeed_title
# ROOT_DIR=/home/nayeon/misinfo/results
# SAVE_DIR=$ROOT_DIR/$TASK_NAME/
# DATA_DIR=/home/nayeon/misinfo/data 

# MT_MODEL_PATH=/home/nayeon/misinfo/results/basil_detection,basil_type,basil_polarity,webis,clickbait,rumour_veracity/lr.5e-06_bz.1_dropout.0_decay.0.0_opt.adamw_stopRemove.False_use_ne_text.False_mtloss.1.0-0.1_bz.fit_main_bs.main_focus_lossf.xentropy_gamma.0.5_evalmetric_f1_mt_joint_loss_acc16
# bsz=1
# steps=1

# FEWSHOT=0.1
# CUDA=0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py \
# --model_type roberta \
# --model_name_or_path $MT_MODEL_PATH/checkpoint-2000 \
# --task_name $TASK_NAME \
# --do_train \
# --do_test \
# --do_lower_case \
# --data_dir '' \
# --tokenizer_name 'roberta-large' \
# --max_seq_length 128 \
# --per_gpu_eval_batch_size=$bsz   \
# --per_gpu_train_batch_size=$bsz   \
# --learning_rate 1e-6 \
# --num_train_epochs 5.0 \
# --output_dir $SAVE_DIR \
# --main_task_w 1 \
# --aux_task_w 0.1 \
# --patience 5 \
# --evaluate_during_training \
# --root_dir $DATA_DIR \
# --b_scheduling main_focus \
# --focal_gamma 0.5 \
# --gradient_accumulation_steps $steps \
# --fewshot_train $FEWSHOT \
# --custom_exp_name mt_joint_loss_acc16_then_ST$TASK_NAME$FEWSHOT \
# --overwrite_output_dir