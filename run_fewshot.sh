
echo "========================few shot on FakeNewsNet Dataset {after MT}================================="

# TODO change parameters/experiment-settings to suit your needs.
ROOT_DIR=/home/nayeon

BSZ=1
STEPS=1
FEWSHOT=0.1
LR=5e-6 
EPOCHS=10
PATIENCE=5

CUDA=6


M2_MODEL_PATH=${ROOT_DIR}/misinfo/pretrained/unifiedM2  # for using pretrained version
# M2_MODEL_PATH=PATH/TO/YOUR/M2_MODEL # for using your own

for TASK_NAME in propaganda fnn_buzzfeed fnn_politifact fnn_buzzfeed_title
do
    SAVE_DIR=${ROOT_DIR}/misinfo/results/${TASK_NAME}/
    DATA_DIR=${ROOT_DIR}/misinfo/data 

    CUSTOM_NAME=custom-name-for-your-experiment-setting
    LOG_NAME=log_file_name.log

    for FEWSHOT in 0.01 0.05 0.1 0.5 1.0
    do
        CUDA_VISIBLE_DEVICES=${CUDA} python main.py \
        --model_type roberta \
        --model_name_or_path ${M2_MODEL_PATH} \
        --task_name ${TASK_NAME} \
        --do_train \
        --do_test \
        --do_lower_case \
        --tokenizer_name 'roberta-large' \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=${BSZ} \
        --per_gpu_train_batch_size=${BSZ}  \
        --learning_rate ${LR} \
        --num_train_epochs ${EPOCHS} \
        --output_dir ${SAVE_DIR} \
        --patience ${PATIENCE} \
        --evaluate_during_training \
        --data_dir ${DATA_DIR} \
        --gradient_accumulation_steps ${STEPS} \
        --fewshot_train_ratio ${FEWSHOT} \
        --custom_exp_name ${CUSTOM_NAME} \
        --overwrite_output_dir  \
        --log_path ${LOG_NAME}

    done

done