## On Unifying Misinformation Detection

Code for our "On Unifying Misinformation Detection", NAACL2021, https://arxiv.org/pdf/2104.05243.pdf 

#### Setting Up Environment: 

1. `pip install -r requirement.txt` to install all the dependencies.
2. From https://drive.google.com/drive/folders/1QRBKuzykREarcsn1iaNr5xKSzTGTRwu1?usp=sharing, download:
   - preprocessed.zip, unzip it and place inside `ROOTDIR/misinfo/data` directory. As result, you will have `ROOTDIR/misinfo/data/preprocessed`
   - pretrained models (unifiedM2.zip and unifiedM2_pheme.zip), unzip and place inside  `ROOTDIR/misinfo/pretrained` directory.
     - unifiedm2.zip: pretrained UnifiedM2 model
     - unifiedM2_pheme.zip: UnifiedM2 that is trained without rumor dataset to test for PHEME's leave-one-event-out setting 

 

#### Code Structure:

* **/data_utils**: contains codes data preprocessing and data analysis 
* **/model**: contains codes for ST-roberta and UnifiedM2 
* **/utils**: contains util codes for training/evaluating our models



#### How to run:

Refer to bash scripts listed below for your intended experiment (Before running the code, modify the arguments/parameters that fits your need/setting):

* `bash run_mt.sh` : for training the shared UnifiedM2 encoder

* `bash run_ft.sh` : for fine-tuning UnifiedM2 encoder to obtain better performance on individual tasks
* `bash run_fewshot.sh` : for few-shot experiment (Section 4.1)

* `bash run_pheme_leave_one_event_out.sh` : for leave-one-event-out experiment setting for PHEME dataset (Section 4.2)



#### Other details:

To add your custom tasks, made adjustments in:

* `datasets.py` - modify `get_processor()` and `get_misinfo_datset()` function,  and add a data processor by following the format of other data processors
* `utils/const.py` - add your custom task name to the const file
* For other questions, send email to nyleeaa@connect.ust.hk # UnifiedM2
