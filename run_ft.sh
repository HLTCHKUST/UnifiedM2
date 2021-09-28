
echo "==============================STEP 2: Train ST on top of MT encoder by fine-tuning=============================="

# TODO change parameters/experiment-settings to suit your needs.
ROOT_DIR=/home/nayeon

CUDA=0
TASK_NAME=webis

DATA_DIR=${ROOT_DIR}/misinfo/data 
BSZ=8
STEPS=4

LR=5e-6
EPOCHS=15
PATIENCE=5

M2_MODEL_PATH=${ROOT_DIR}/misinfo/pretrained/unifiedM2  # for using pretrained version
# M2_MODEL_PATH=PATH/TO/YOUR/M2_MODEL # for using your own

SAVE_DIR=${ROOT_DIR}/misinfo/results/${TASK_NAME}

CUSTOM_NAME=custom-name-for-your-experiment-setting

CUDA_VISIBLE_DEVICES=${CUDA} python main.py \
    --model_type roberta \
    --model_name_or_path ${M2_MODEL_PATH} \
    --task_name ${TASK_NAME} \
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





