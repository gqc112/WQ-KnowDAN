#!/bin/bash

timestamp=`date "+%d.%m.%Y_%H.%M.%S"`
output_dir='./logs/'
config_file='./configs/CoNLL04/bio_config'
config_adv_file='./configs/CoNLL04/bio_config_adv'

# unzip the embeddings file 解压缩嵌入文件
unzip data/CoNLL04/vecs.lc.over100freq.zip -d data/CoNLL04/

mkdir -p $output_dir

#train on the training set and evaluate on the dev set to obtain early stopping epoch
#在训练集上训练并在验证集上评估以获得早停周期
#python -u train_es.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.dev_${timestamp}.txt

python -u train_es.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.dev_${timestamp}.txt


##python -u dev_.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.dev_${timestamp}.txt

#train on the train and dev sets and evaluate on the test set until (1) max epochs limit exceeded or
#(2) the limit specified by early stopping after executing train_es.py
#在连接的（训练+验证）集合上训练并在测试集合上评估，直到（1） 最大周期或（2） 超过早停限制（由y train_es.py指定）

#python -u train_eval.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.test.${timestamp}.txt