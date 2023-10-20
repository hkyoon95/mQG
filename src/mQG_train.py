import os
from os.path import join

cur_dir = os.path.dirname(os.path.realpath(__file__))

beta = 1.0
attribute = 'mQG'
ori_data_dir = '{}/data/ftqa_data.pkl'.format(cur_dir)
reload_data=True
loss_n = 'ce_mqs'
data_check_dir = '{}/data1/{}'.format(cur_dir, attribute)
output_dir='{}/checkpoint/{}/output_dir/'.format(cur_dir, attribute)
model_name_or_path = 'facebook/bart-large'
tokenizer_name = 'facebook/bart-large'
config_name = 'facebook/bart-large'
gpus=1
max_epochs=4
learning_rate=5e-6
train_batch_size=4
max_source_length=512
max_target_length=128
val_max_target_length=128
test_max_target_length=128
eval_batch_size=4
cache_dir='pretrained'
path_or_data='data'

cmd_str = 'python {} \
    --data_dir={} \
    --model_name_or_path={} \
    --tokenizer_name={} \
    --config_name={} \
    --do_train \
    --gpus={} \
    --max_epochs={} \
    --learning_rate={} \
    --train_batch_size={} \
    --max_target_length={} \
    --val_max_target_length={} \
    --test_max_target_length={} \
    --max_source_length={} \
    --eval_batch_size={} \
    --cache_dir={} \
    --output_dir={} \
    --attribute={} \
    --ori_dir={} \
    --beta={} \
    --loss_n={} \
    --reload_data={} \
    --path_or_data={}'.format(cur_dir+'/1_train.py', 
                            data_check_dir, 
                            model_name_or_path, 
                            tokenizer_name, 
                            config_name,
                            gpus,
                            max_epochs,
                            learning_rate,
                            train_batch_size,
                            max_target_length,
                            val_max_target_length,
                            test_max_target_length,
                            max_source_length,
                            eval_batch_size,
                            cache_dir,
                            output_dir,
                            attribute,
                            ori_data_dir,
                            beta,
                            loss_n,
                            reload_data,
                            path_or_data,
                            )

os.system(cmd_str)