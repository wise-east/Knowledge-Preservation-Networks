# Knowledge-Preservation-Networks
This the code of the paper 'Domain-Lifelong Learning for Dialogue State Tracking via Knowledge Preservation Networks'.
Python=3.8.5
torch=1.5.0
transformers=3.2.0
numpy=1.19.4


Steps: 
1. `create_data.py`: downloads MultiWOZ and splits it by train/valid/test
1. `create_lifelong_data21.py`: split data by domain for continual learning set up 
1. `train_test.py`: run training for Knowledge Preservation Network (KPN)
