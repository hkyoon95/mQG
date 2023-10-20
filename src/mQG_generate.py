import os
import glob

cur_dir = os.path.dirname(os.path.realpath(__file__)) 

attribute = 'mQG'
data = 'test'
search_mode = 'beam'
data_dir = '{}/data'.format(cur_dir)
model_name_or_path = 'facebook/bart-large'
tokenizer_name = 'facebook/bart-large'
config_name = 'facebook/bart-large'
input_path = None
bs=16
device=2
cache_dir='pretrained'
generate_number = 4
prefix = 2
beam = 5

# path for results
output_dir='{}/output/result'.format(cur_dir)
test_max_target_length=64

ckpt_path = '{}/checkpoint/mQG_epoch=13.ckpt'.format(cur_dir)

cmd_str = 'python {} \
    --model_name_or_path={} \
    --tokenizer_name={} \
    --config_name={} \
    --ckpt_path={} \
    --input_path={} \
    --bs={} \
    --device={} \
    --cache_dir={} \
    --output_dir={} \
    --search_mode={} \
    --data={} \
    --beam={} \
    --data_dir={} \
    --generate_num={} \
    --prefix={} \
    --test_max_target_length={}'.format(cur_dir+'/2_generate.py',  
                                        model_name_or_path, 
                                        tokenizer_name, 
                                        config_name,
                                        ckpt_path,
                                        input_path,
                                        bs,
                                        device,
                                        cache_dir,
                                        output_dir,
                                        search_mode,
                                        data,
                                        beam,
                                        data_dir,
                                        generate_number,
                                        prefix,
                                        test_max_target_length,
                                        )
    
os.system(cmd_str)