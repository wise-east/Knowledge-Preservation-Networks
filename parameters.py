class _Config:
    def __init__(self):
        self.use_cuda = True

    def init_handler(self, m):
        init_method = {
            'MultiWOZ21': self._MultiWOZ21_init,
            'SGD': self._SGD,
        }
        init_method[m]()

    def _MultiWOZ21_init(self):
        self.raw_data_path = 'data/MultiWOZ_2.1'
        self.data_path = 'data/MultiWOZ_2.1/lifelong'
        

        self.bert_base_uncased_path = 'bert-base-uncased'
        self.bert_config_path = 'bert-base-uncased'

        self.dialog_turn_num = 1
        self.per_epoch_all = 30
        self.per_epoch_list = [30, 30, 10, 30, 30, 30, 30, 30, 30, 30]
        self.shuffle = True
        self.dropout = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.encoder_lr = 5e-5
        self.decoder_lr = 5e-5
        self.batch_size = 4
        self.test_batch_size = 4
        self.truth_belief_state = True
        self.teacher_force = 0.5
        self.decoder_type = 'gru'
        self.max_r_len = 12
        self.max_seq_length = 256
        self.memory_num = 50  # 0 50 
        self.alpha_0 = 0.1
        self.alpha_1 = 0.2
        self.temperature = 2
        self.reverse_type = 'KPN'  # 'none'  'full'  'KPN'
        self.knowledge_type = 'KPN'  # 'none'  'KPN'
        self.increment_dev_set = False # False True 
        self.multitask = False # False True 
        self.multitask_all = True 

        self.model_save = f'save/MultiWOZ_2.1/ep{self.per_epoch_all}_increment_dev:{self.increment_dev_set}_kt:{self.knowledge_type}_mem:{self.memory_num}_multitask:{self.multitask}/'
        if self.multitask_all: 
            self.memory_num = 0 
            self.reverse_type = "none" 
            self.knowledge_type = "none" 
            self.model_save = f'save/MultiWOZ_2.1/ep{self.per_epoch_all}_multitask_all'

        self.rkd_filter_none = False  # True [False is better]

    def _SGD(self):
        self.raw_data_path = 'data/SGD/train'
        self.data_path = 'data/SGD/lifelong'
        self.model_save = 'save/SGD/'

        self.bert_base_uncased_path = '/data/liuqbdata/transformers/bert/bert-base-uncased/'
        self.bert_config_path = '/data/liuqbdata/transformers/bert/bert-base-uncased/config.json'

        self.dialog_turn_num = 1
        self.per_epoch_all = 30
        self.per_epoch_list = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
        self.shuffle = True
        self.dropout = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        self.encoder_lr = 5e-5
        self.decoder_lr = 5e-5
        self.batch_size = 3
        self.test_batch_size = 3
        self.truth_belief_state = True
        self.teacher_force = 0.5
        self.decoder_type = 'gru'
        self.max_r_len = 12
        self.max_seq_length = 256
        self.memory_num = 50
        self.alpha_0 = 0.1 # 0.25
        self.alpha_1 = 0.2
        self.temperature = 2
        self.reverse_type = 'KPN'  # 'none'  'full'  'KPN'
        self.knowledge_type = 'KPN'  # 'none'  'KPN'

        self.rkd_filter_none = True  # True False [True is better]
global_parm = _Config()