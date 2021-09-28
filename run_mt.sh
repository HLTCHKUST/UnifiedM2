

echo "============================== STEP1: Train MT first - train shared encoder =============================="

# TODO change parameters/experiment-settings to suit your needs.
ROOT_DIR=/home/nayeon

CUDA=0
LR=5e-6
EPOCHS=15
PATIENCE=5
BSZ=8
STEPS=4
CUSTOM_NAME=custom-name-for-your-experiment-setting

TASK_NAMES=basil_detection,basil_type,basil_polarity,webis,clickbait,rumour_veracity_binary
SAVE_DIR=${ROOT_DIR}/misinfo/results/${TASK_NAMES}/
DATA_DIR=${ROOT_DIR}/misinfo/data 



CUDA_VISIBLE_DEVICES=$CUDA python main.py \
--model_type roberta_mt \
--model_name_or_path roberta-large \
--task_name ${TASK_NAMES} \
--do_train \
--do_test \
--do_lower_case \
--tokenizer_name 'roberta-large' \
--max_seq_length 128 \
--per_gpu_eval_batch_size=${BSZ}   \
--per_gpu_train_batch_size=${BSZ}   \
--learning_rate ${LR} \
--num_train_epochs ${EPOCHS} \
--output_dir ${SAVE_DIR} \
--patience ${PATIENCE} \
--evaluate_during_training \
--data_dir ${DATA_DIR} \
--gradient_accumulation_steps ${STEPS} \
--custom_exp_name ${CUSTOM_NAME} \
--overwrite_output_dir
