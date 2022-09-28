from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import pandas as pd
from loss import CosineSimilarityLoss

# config
model_name = 'bert-base-chinese'
train_batch_size =4 
num_epochs = 4
model_save_path = 'wmdtest_output'
logging.basicConfig(format='%(asctime)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[LoggingHandler()])

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
  pooling_mode_mean_tokens=True,
  pooling_mode_cls_token=False,
  pooling_mode_max_tokens=False)
# 相当于pytorch nn Module中的Sequence
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# preprocess dataset
train_samples = []
dev_samples = []
test_samples = []
def load(path):
  df = pd.read_csv(path)
  samples = []
  for idx,item in df.iterrows():
    samples.append(InputExample(texts=[item['sentence1'], item['sentence2']], label=float(item['label'])))
  return samples

train_samples = load('D:/1.Workstation/VSCodeProject/semantic-similarity-main/semantic-similarity-main/bert_wmd/data/train.csv')
test_samples = load('D:/1.Workstation/VSCodeProject/semantic-similarity-main/semantic-similarity-main/bert_wmd/data/test.csv')
dev_samples = load('D:/1.Workstation/VSCodeProject/semantic-similarity-main/semantic-similarity-main/bert_wmd/data/dev.csv')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# 损失函数
train_loss = CosineSimilarityLoss(model=model)
# 评估
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
  evaluator=evaluator,
  epochs=num_epochs,
  evaluation_steps=1000,
  warmup_steps=warmup_steps,
  output_path=model_save_path)

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='ant-test')
test_evaluator(model, output_path=model_save_path)