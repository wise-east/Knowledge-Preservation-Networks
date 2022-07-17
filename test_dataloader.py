# examine each sample from dataloader 

from parameters import global_parm as par
from read_MultiWOZ21 import increment_dataset, get_schema, DOMAINS, prepare_dataset, data_tokenizer_loader, decode_belief_state, predicts_to_list
from utils.basic_func import read_json
import os 
from transformers import BertTokenizer
from train_test import DST_model
import torch 
from pprint import pprint 


def test_dataloader(): 
    dataset="MultiWOZ21"
    thread_num=0
    mode="train"
    par.init_handler(dataset)
    par.thread_num = thread_num
    par.mode = mode

    n_gpu = torch.cuda.device_count()
    tokenizer = BertTokenizer.from_pretrained(par.bert_base_uncased_path)
    current_DST_model = DST_model(n_gpu, tokenizer.convert_tokens_to_ids(['[PAD]'])[0])

    previous_domains, data_memory, last_model = [], {}, None
    for per_domain_idx, per_domain in enumerate(DOMAINS): 
        previous_domains += per_domain.split('-')

        current_schema = get_schema(par, set(previous_domains))
        data_memory_samples = []
        # load data for new domain 
        train_data_raw = prepare_dataset(par= par,
                                        data=read_json(os.path.join(par.data_path, per_domain+'[train.json')) +
                                            data_memory_samples,
                                        domain=per_domain,
                                        schema=current_schema,
                                        tokenizer=tokenizer)
        train_data_loader = data_tokenizer_loader(par, train_data_raw, tokenizer, current_schema, par.shuffle, True)
        example = train_data_raw[0][1]
        
        print(example)
        break 

if __name__ == "__main__": 
    test_dataloader() 
        
        
