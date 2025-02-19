import os
import argparse
from parameters import global_parm as par
from utils.basic_model import masked_cross_entropy_for_value, MultinomialKLDivergenceLoss
from transformers import BertTokenizer, BertConfig, AdamW
from read_MultiWOZ21 import increment_dataset, get_schema, DOMAINS, prepare_dataset, data_tokenizer_loader, \
    decode_belief_state, predicts_to_list
from utils.basic_func import read_json
from bert_model import Bert_DST
import torch
from evaluator import evaluate
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from copy import deepcopy
import time
import json 
from pathlib import Path 
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
 


KEY_RESULTS = {} 

class DST_model(object):
    def __init__(self, n_gpu, pad_idx):
        super().__init__()
        # self.model_config = BertConfig.from_json_file(par.bert_config_path)
        self.model_config = BertConfig.from_pretrained(par.bert_config_path)
        self.model_config.dropout = par.dropout
        self.model_config.attention_probs_dropout_prob = par.attention_probs_dropout_prob
        self.model_config.hidden_dropout_prob = par.hidden_dropout_prob

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_dst_model = Bert_DST(par, self.model_config, pad_idx)

        self.bert_dst_model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
        self.bert_dst_model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
        self.bert_dst_model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)

        self.bert_dst_model.to(self.device)

        no_decay = [ 'bias', 'LayerNorm.bias', 'LayerNorm.weight' ]
        enc_param_optimizer = list(self.bert_dst_model.encoder.named_parameters())
        enc_optimizer_grouped_parameters = [
            {'params': [ p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay) ],
             'weight_decay': 0.01},
            {'params': [ p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay) ], 'weight_decay': 0.0}
        ]
        self.enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=par.encoder_lr)
        self.dec_optimizer = AdamW(list(self.bert_dst_model.decoder.parameters()), lr=par.decoder_lr)

        if n_gpu>1:
            self.bert_dst_model = torch.nn.DataParallel(self.bert_dst_model)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.distill_criterion = nn.CosineEmbeddingLoss()
        self.dst_evaluate = evaluate()

        self.means, self.precision_matrices, self.domain_prototype = None, None, None

    def train(self, dataloader, schema, tokenizer, dev_dataloader, current_domain, per_domain_idx, last_model, data_memory):
        self.bert_dst_model.train()

        best_joint_goal_accuracy = 0
        update_step = 0 
        for per_epoch in range(par.per_epoch_list[per_domain_idx]):
            data_iterator, dialog_batch_num, train_loss,train_loss_all = dataloader.mini_batch_iterator(), 0, 0, [0, 0]
            for batch_dialog in data_iterator:
                predict_belief_state = None
                dialog_batch_num += 1
                # does this mean that batch size = 1? No
                # each batch_turn is a batch of turns from different conversations that share the same turn_idx
                # so the batch_size should most of the time = 4 unless dataset is very small or for conversations with very many turns 
                for batch_turn in batch_dialog:
                    batch_turn_data, max_update, max_value = dataloader.fill_belief_state(
                        batch_turn, predict_belief_state, self.device, self.bert_dst_model.training)

                    predicts_g, predicts_y, predicts_state = self.bert_dst_model(batch_turn_data, schema,
                                                                                 [max_update, max_value],
                                                                                 is_update_ewc=False)
                    predicts = [predicts_g, predicts_y]

                    loss, loss_all = self.calculate_loss(predicts, batch_turn_data, tokenizer, max_update)                    
                    
                    train_loss += loss.item()
                    train_loss_all[0] += loss_all[0].item()
                    if loss_all[1] != 0:
                        train_loss_all[1] += loss_all[1].item()

                    if (par.knowledge_type in ['KPN']) & (last_model is not None):
                        with torch.no_grad():
                            predicts_g_last, predicts_y_last, predicts_state_last = last_model(batch_turn_data,
                                                                                               schema,
                                                                                               [max_update,
                                                                                                max_value],
                                                                                               is_update_ewc=False)
                        temp_predicts_g = F.softmax(predicts_g/par.temperature, dim=2)
                        temp_predicts_g_last = F.softmax(predicts_g_last/par.temperature, dim=2)
                        loss_kl_gate = torch.log(temp_predicts_g)*temp_predicts_g_last
                        loss_kl_gate = -torch.sum(loss_kl_gate)/(loss_kl_gate.shape[0]*loss_kl_gate.shape[1])

                        loss = loss + par.alpha_0 * loss_kl_gate
                        if max_update != 0:
                            loss_kl_y = MultinomialKLDivergenceLoss(predicts_y, predicts_y_last)
                            loss = loss + par.alpha_0 * loss_kl_y

                        batch_size, slot_size, hidden_size = predicts_state[1].shape[0],\
                                                             predicts_state[1].shape[1],\
                                                             predicts_state[1].shape[2]
                        loss_encoder_slot = self.distill_criterion(predicts_state[1].reshape(-1, hidden_size),
                                                                   predicts_state_last[1].reshape(-1, hidden_size),
                                                                   torch.ones(batch_size*slot_size).cuda())
                        batch_size, slot_size, value_size, hidden_size = predicts_state[2].shape[0],\
                                                                         predicts_state[2].shape[1],\
                                                                         predicts_state[2].shape[2],\
                                                                         predicts_state[2].shape[3]
                        loss_decoder_hidden = self.distill_criterion(predicts_state[2].reshape(-1, hidden_size),
                                                                   predicts_state_last[2].reshape(-1, hidden_size),
                                                                   torch.ones(batch_size*slot_size*value_size).cuda())

                        loss = loss + par.alpha_1 * (loss_encoder_slot + loss_decoder_hidden)

                    if not par.truth_belief_state:
                        predicts_list = predicts_to_list(predicts)
                        predict_belief_state = decode_belief_state(par, predicts_list, tokenizer, schema,
                                                                   predict_belief_state)

                    loss.backward()
                    self.enc_optimizer.step()
                    self.dec_optimizer.step()
                    self.bert_dst_model.zero_grad()
                    update_step += 1 

                if dialog_batch_num % 40 == 0:
                    logger.info(f'current batch number: {dialog_batch_num}\t train loss: {train_loss:.4f} {[round(tl, 4) for tl in train_loss_all]}')
                    writer.add_scalar('Loss/train', loss, update_step)
                    writer.add_scalar('Loss_gate/train', loss_all[0], update_step)
                    writer.add_scalar('Loss_generation/train', loss_all[1], update_step)

                    train_loss, train_loss_all = 0, [0, 0]
                    
            if per_epoch % 1 == 0:
                scores, val_loss, val_loss_gate, val_loss_gen = self.validate(dev_dataloader, schema, tokenizer)
                writer.add_scalar('Loss/val', val_loss, update_step)
                writer.add_scalar('Loss_gate/val', val_loss_gate, update_step)
                writer.add_scalar('Loss_generation/val', val_loss_gen, update_step)
                
                joint_goal_accuracy = self.get_print_score(scores, per_epoch)
                if joint_goal_accuracy > best_joint_goal_accuracy:
                    logger.info(f"New best validation set joint goal accuracy: {joint_goal_accuracy} >> {best_joint_goal_accuracy}")
                    logger.info('saved models')
                    best_joint_goal_accuracy = joint_goal_accuracy
                    self.save_model(per_epoch, par.result_path+current_domain)
                else: 
                    logger.info(f"Didn't beat best validation set joint goal accuracy: {joint_goal_accuracy} << {best_joint_goal_accuracy}")

    def validate(self, dataloader, schema, tokenizer):
        self.bert_dst_model.eval()
        data_iterator, scores = dataloader.mini_batch_iterator(), []
        
        loss_list, loss_gate, loss_gen = [], [], []  
        for batch_dialog in data_iterator:
            predict_belief_state = None
            for batch_turn in batch_dialog:
                batch_turn_data, max_update, max_value = dataloader.fill_belief_state(
                    batch_turn, predict_belief_state, self.device, self.bert_dst_model.training)

                predicts_g, predicts_y, _ = self.bert_dst_model(batch_turn_data, schema, [ max_update, max_value ],
                                                                is_update_ewc=False)
                predicts = [ predicts_g, predicts_y ]

                # loss, loss_all = self.calculate_loss(predicts, batch_turn_data, tokenizer, max_update)
                # loss_list.append(loss) 
                # loss_gate.append(loss_all[0])
                # loss_gen.append(loss_all[1])

                predicts_list = predicts_to_list(predicts)
                predict_belief_state = decode_belief_state(par, predicts_list, tokenizer, schema, predict_belief_state)
                batch_scores = self.dst_evaluate.compare_acc(predict_belief_state, batch_turn)
                if len(scores) == 0:
                    scores = batch_scores
                else:
                    for per_score_idx, per_score in enumerate(batch_scores):
                        scores[per_score_idx] += per_score
        self.bert_dst_model.train()
        return scores, np.mean(loss_list), np.mean(loss_gate), np.mean(loss_gen)

    def update_data_memory(self, data_memory, update_data_raw, update_data_loader, domain, domain_idx, schema, tokenizer):
        if par.reverse_type == 'full':
            data_memory[ domain ] = update_data_raw
        elif par.reverse_type == 'none':
            data_memory[ domain ] = [ ]
        else:
            per_domain_num = int(par.memory_num / (domain_idx + 1))
            for sv in data_memory.items():
                data_memory[sv[0]] = sv[1][:per_domain_num]

            if par.reverse_type == 'KPN':
                prototypes, prototypes_all = self.get_prototype(update_data_loader, schema, tokenizer, pro_type='domain-slot')
                prototypes = np.concatenate(prototypes, axis=0)
                current_prototype = np.mean(prototypes, axis=0)

                temp_domain, current_prototype_all = domain.split('-'), []
                for s_idx, s in enumerate(schema):
                    temp_d = s.split('-')[0]
                    if temp_d in temp_domain:
                        current_prototype_all.append(current_prototype[s_idx])
                    else:
                        current_prototype_all.append(None)

                distance_all, slot_num, prototypes_all = [], len(schema), np.concatenate(prototypes_all, axis=0)
                for per_sample in range(prototypes.shape[0]):
                    per_sample_distance = []
                    for temp_s_idx in range(slot_num):
                        if current_prototype_all[temp_s_idx] is not None:
                            temp_distance = np.sqrt(np.sum(np.power(prototypes_all[per_sample][temp_s_idx] -
                                                                    current_prototype_all[temp_s_idx], 2)))
                            per_sample_distance.append(temp_distance)
                    distance_all.append([per_sample, sum(per_sample_distance)])

                data_memory[domain] = []
                distance_all = sorted(distance_all, key=lambda x:x[1])
                for sample_idx in distance_all[:per_domain_num]:
                    data_memory[domain].append(update_data_raw[sample_idx[0]])

        return data_memory

    def get_prototype(self, dataloader, schema, tokenizer, pro_type='domain'):
        self.bert_dst_model.eval()
        data_iterator, prototype, prototype_all = dataloader.mini_batch_iterator(), [ ], [ ]
        for batch_dialog in data_iterator:
            predict_belief_state = None
            turn_prototype, turn_prototype_all = [], []
            for batch_turn in batch_dialog:
                batch_turn_data, max_update, max_value = dataloader.fill_belief_state(
                    batch_turn, predict_belief_state, self.device, self.bert_dst_model.training)

                predicts_g, predicts_y, states = self.bert_dst_model(batch_turn_data, schema, [ max_update, max_value ],
                                                                     is_update_ewc=False)
                predicts = [ predicts_g, predicts_y ]

                predicts_list = predicts_to_list(predicts)
                predict_belief_state = decode_belief_state(par, predicts_list, tokenizer, schema, predict_belief_state)

                if pro_type == 'domain':
                    turn_prototype.append(states[0].tolist())
                elif pro_type == 'domain-slot':
                    temp_pro = np.array(states[1].tolist())
                    turn_prototype_all.append(temp_pro.copy())
                    if par.rkd_filter_none:
                        for pro_per_batch_idx, pro_per_batch in enumerate(batch_turn):
                            temp_belief_state = dict(pro_per_batch['belief_state'])
                            for s_idx, s in enumerate(schema):
                                v = temp_belief_state.get(s)
                                if v is None:
                                    temp_pro[pro_per_batch_idx][s_idx] = 0
                    turn_prototype.append(temp_pro)
            turn_prototype = np.mean(np.array(turn_prototype), axis=0)
            prototype.append(turn_prototype)
            if pro_type == 'domain-slot':
                prototype_all.append(np.mean(np.array(turn_prototype_all), axis=0))
        self.bert_dst_model.train()
        if pro_type == 'domain-slot':
            return prototype, prototype_all
        else:
            return prototype

    def calculate_loss(self, predicts, batch_turn_data, tokenizer, max_update):
        gate_loss = self.criterion_ce(predicts[0].view(-1, 4), batch_turn_data[4].view(-1))
        generation_loss = 0
        if max_update == 0:
            loss = gate_loss
        else:
            generation_loss = masked_cross_entropy_for_value(predicts[1].contiguous(),
                                                             batch_turn_data[5].contiguous(),
                                                             tokenizer.vocab['[PAD]'])
            loss = gate_loss + generation_loss
        return loss, [gate_loss, generation_loss]

    def save_model(self, epoch, path=None):
        all_state = {'us': self.bert_dst_model.state_dict(),
                     'config': par.__dict__,
                     'epoch': epoch}
        
        with open(path+par.thread_num+'.pkl', 'wb') as f:
            torch.save(all_state, f)

    def load_model(self, path=None):
        all_state = torch.load(path, map_location=self.device)
        logger.info(f"loaded from epoch {all_state['epoch']}")
        self.bert_dst_model.load_state_dict(all_state['us'], strict=False)

    @staticmethod
    def get_print_score(scores, epoch):
        joint_goal_accuracy = scores[ 0 ] / scores[ 1 ]

        logger.info(f' epoch: {epoch} joint goal accuracy: {joint_goal_accuracy:.4f}')
        return joint_goal_accuracy

def main():
    tokenizer = BertTokenizer.from_pretrained(par.bert_base_uncased_path)
    current_DST_model = DST_model(n_gpu, tokenizer.convert_tokens_to_ids(['[PAD]'])[0])

    previous_domains, data_memory, last_model = [], {}, None
    for per_domain_idx, per_domain in enumerate(DOMAINS):
        previous_domains += per_domain.split('-')
        
        if par.multitask_all: 
            previous_domains= [] 
            for dom in DOMAINS: 
                previous_domains += dom.split('-')
                
        current_schema = get_schema(par, set(previous_domains))

        data_memory_samples = []
        for samples in data_memory.items():
            data_memory_samples += samples[1]

        # load data for new domain 
        if par.get_upperbound: 
            train_data_list = increment_dataset(par= par, domains=DOMAINS[:per_domain_idx+1], data_type='train')
        elif par.multitask_all: 
            # train with all 
            train_data_list = increment_dataset(par= par, domains=DOMAINS, data_type='train')
        else: 
            train_data_list = read_json(os.path.join(par.data_path, per_domain+'[train.json')) + data_memory_samples
        train_data_raw = prepare_dataset(par= par,
                                        data=train_data_list,
                                         domain=per_domain,
                                         schema=current_schema,
                                         tokenizer=tokenizer)
        train_data_loader = data_tokenizer_loader(par, train_data_raw, tokenizer, current_schema, par.shuffle, True)

        # expand dev set to include new domain
        if par.increment_dev_set or par.get_upperbound: 
            dev_data_list = increment_dataset(par= par, domains=DOMAINS[:per_domain_idx+1], data_type='dev')
        elif par.multitask_all: 
            dev_data_list = increment_dataset(par= par, domains=DOMAINS, data_type='dev')
        else: 
            dev_data_list = increment_dataset(par= par, domains=[per_domain], data_type='dev')
        dev_data_raw = prepare_dataset(par= par,
                                       data=dev_data_list,
                                       domain=per_domain,
                                       schema=current_schema,
                                       tokenizer=tokenizer)
        dev_data_loader = data_tokenizer_loader(par, dev_data_raw, tokenizer, current_schema, False, False)

        # expand test set to include new domain 
        if par.multitask_all: 
            test_data_list = increment_dataset(par= par, domains=DOMAINS, data_type='test')
        else: 
            test_data_list = increment_dataset(par= par, domains=DOMAINS[:per_domain_idx+1], data_type='test')
        
        test_data_raw = prepare_dataset(par= par,
                                        data=test_data_list,
                                        domain=per_domain,
                                        schema=current_schema,
                                        tokenizer=tokenizer)
        test_data_loader = data_tokenizer_loader(par, test_data_raw, tokenizer, current_schema, False, False)

        if par.mode == 'train':
            logger.info(f'Train: {per_domain}')
            current_DST_model.train(train_data_loader, current_schema, tokenizer, dev_data_loader, per_domain, per_domain_idx, last_model, data_memory)

        logger.info(f'Test: {per_domain}')
        current_DST_model.load_model(par.result_path+per_domain+par.thread_num+'.pkl')
        scores, test_loss, test_loss_gate, test_loss_gen = current_DST_model.validate(test_data_loader, current_schema, tokenizer)
        _ = current_DST_model.get_print_score(scores, 'test '+per_domain)
        writer.add_scalar("Loss/test", test_loss, per_domain_idx)
        writer.add_scalar("Loss_gate/test", test_loss_gate, per_domain_idx)
        writer.add_scalar("Loss_gen/test", test_loss_gen, per_domain_idx)
        logger.info(f"{per_domain} - Test loss: {test_loss:.4f} | Test all loss: gate - {test_loss_gate:.4f} gen - {test_loss_gen:.4f}")

        KEY_RESULTS[f"{per_domain}_scores"] = scores 

        if par.knowledge_type in ['KPN']:
            last_model = deepcopy(current_DST_model.bert_dst_model)
        else:
            last_model = None

        if par.multitask_all: 
            break 

        if par.mode == 'train': 
            # update memory with prototypical samples 
            logger.info(f'Update Memory: {per_domain}')
            update_data_raw = read_json(os.path.join(par.data_path, per_domain + '[train.json'))
            update_data_prepare = prepare_dataset(par=par,
                                            data=update_data_raw,
                                            domain=per_domain,
                                            schema=current_schema,
                                            tokenizer=tokenizer)
            update_data_loader = data_tokenizer_loader(par, update_data_prepare, tokenizer, current_schema, False, False)
            data_memory = current_DST_model.update_data_memory(data_memory, update_data_raw, update_data_loader,
                                                            per_domain, per_domain_idx, current_schema, tokenizer)
    return 0

if __name__ == '__main__':

    writer = SummaryWriter()

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train') # train test
    parser.add_argument('-thread_num', default='0') # 0
    parser.add_argument('-dataset', default='MultiWOZ21') # MultiWOZ21
    args = parser.parse_args()
    par.init_handler(args.dataset)
    par.thread_num = args.thread_num
    par.mode = args.mode
    
    par.save() 
    logger.add(os.path.join(par.result_path + f"log_{par.mode}.txt"))

    logger.info(f"Config: {par}")
    
    start_local = time.localtime()
    start_time = time.time() 
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', start_local)}")
    random_seed = 40
    np.random.seed(random_seed)
    random.seed(random_seed)
    rng = random.Random(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    n_gpu = torch.cuda.device_count()
    logger.info(f"num gpus: {n_gpu}")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f'this is the thread number {par.thread_num}')
    
    os.makedirs(par.result_path, exist_ok=True)
    main()
    end_time = time.time() 
    end_local = time.localtime()
    logger.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', end_local)}")
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)) 
    logger.info(f"Elapsed time: {elapsed_time}")

    KEY_RESULTS["elapsed_time"] = elapsed_time 
    key_results_path = Path(par.result_path) / "key_results.json"
    with key_results_path.open("w") as f: 
        json.dump(KEY_RESULTS, f, indent=4, sort_keys=True)
    
    