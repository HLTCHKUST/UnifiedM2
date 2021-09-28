echo "============================== PHEME leave-one-event-out setting =============================="

# TODO change parameters/experiment-settings to suit your needs.

ROOT_DIR=/home/nayeon

TASK_NAME=rumour_veracity
CUDA=1
BSZ=8
STEPS=4
LR=5e-6 
EPOCHS=15
PATIENCE=5

DATA_DIR=${ROOT_DIR}/misinfo/data 
SAVE_DIR=${ROOT_DIR}/misinfo/results/emnlp/${TASK_NAME}/

M2_MODEL_PATH=${ROOT_DIR}/misinfo/pretrained/unifiedM2_pheme  # for using pretrained version
# M2_MODEL_PATH=PATH/TO/YOUR/M2_MODEL # for using your own

CUSTOM_NAME=custom-name-for-your-experiment-setting
LOG_NAME=LOG_NAME=log_file_name.log

for TEST_EVENT_NAME in ottawashooting sydneysiege charliehebdo putinmissing gurlitt germanwings ebola prince ferguson
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
        --log_path ${LOG_NAME} \
        --test_event_name ${TEST_EVENT_NAME} \
        --overwrite_output_dir 
done