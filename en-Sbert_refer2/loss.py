import torch
from torch import nn, Tensor, tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
import numpy as np



class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)


    """
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        # print("embeddings.shape:", np.shape(embeddings))
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))

import numpy as np
from scipy.optimize import linprog
from numpy import float64

def batch_wmdistance1(query_vec, question_vec, query, questions):
    """WMD（Word Mover's Distance）
    x.shape=[m,d], y.shape=[n,d]
    """
    q_Q_wmd = []
    batch = len(query)
    for i in range(batch):
        query_len = len(query[i])
        question_len = len(questions[i])
        q_Q_wmd.append(wmdistance(query_vec[i][:query_len], question_vec[i][:question_len])) 
        # 为啥要截断这个操作，后面query_vec和querys的长度难道不一样吗？

    return q_Q_wmd

def batch_wmdistance2(query_vec, question_vec):
    """WMD（Word Mover's Distance）
    x.shape=[m,d], y.shape=[n,d]
    """
    q_Q_wmd = []
    batch = len(query_vec) # xyx
    for i in range(batch):
        # query_len = len(query[i])
        # question_len = len(questions[i])
        q_Q_wmd.append(wmdistance(query_vec[i], question_vec[i])) 
        # 为啥要截断这个操作，后面query_vec和querys的长度难道不一样吗？

    return q_Q_wmd

def wmdistance(sent1, sent2):
    """WMD（Word Mover's Distance）
    x.shape=[m,d], y.shape=[n,d]
    """
    p = np.ones(sent1.shape[0], dtype=np.float64) / sent1.shape[0]
    q = np.ones(sent2.shape[0], dtype=np.float64) / sent2.shape[0]
    D = float64(np.sqrt(np.square(sent1[:, None] - sent2[None, :]).mean(axis=2)))
    return wasserstein_distance(p, q, D)


def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun


class WMDSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)


    """
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # print("embeddings.shape:",embeddings.shape)
        # output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        output = self.cos_score_transformation(tensor(batch_wmdistance2(embeddings[0],embeddings[1])))
        
        return self.loss_fct(output, labels.view(-1))

