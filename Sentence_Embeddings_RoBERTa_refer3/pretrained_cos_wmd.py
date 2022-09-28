#-*- coding: utf-8 -*-

import torch
from transformers import RobertaModel, RobertaTokenizer, BertTokenizer, BertModel  #RobertaModel, RobertaTokenizer 
import sys
import re
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn import manifold          #use this for MDS computation
import pandas as pd
import numpy as np

#visualization libs
import plotly
plotly.io.orca.config.executable = '/usr/local/anaconda3/envs/xyx-py36-trf302/bin/orca'
plotly.io.orca.config.save()

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文乱码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']

# % matplotlib inline

#Used to calculation of word movers distance between sentence
from collections import Counter

#Library to calculate Relaxed-Word Movers distance
from wmd import WMD
from wmd import libwmdrelax

# error: package typing-extensions requires python's version >= 3.7

#Define some constants
# PRETRAINED_MODEL = 'roberta-base'     #'bert-large-uncased'
PRETRAINED_MODEL = 'hfl/chinese-roberta-wwm-ext'     #'bert-large-uncased'

MAX_LEN = 512

#Initialize tokenizer
# tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)    #BertTokenizer.from_pretrained(PRETRAINED_MODEL)
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)    #BertTokenizer.from_pretrained(PRETRAINED_MODEL)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, tokenizer_obj, max_len=MAX_LEN):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    @return   attention_masks_without_special_tok (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model excluding the special tokens (CLS/SEP)
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer_obj.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            truncation=True,              #Truncate longer seq to max_len
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    
    #lets create another mask that will be useful when we want to average all word vectors later
    #we would like to average across all word vectors in a sentence, but excluding the CLS and SEP token
    #create a copy
    attention_masks_without_special_tok = attention_masks.clone().detach()
    
    #set the CLS token index to 0 for all sentences 
    attention_masks_without_special_tok[:,0] = 0

    #get sentence lengths and use that to set those indices to 0 for each length
    #essentially, the last index for each sentence, which is the SEP token
    sent_len = attention_masks_without_special_tok.sum(1).tolist()

    #column indices to set to zero
    col_idx = torch.LongTensor(sent_len)
    #row indices for all rows
    row_idx = torch.arange(attention_masks.size(0)).long()
    
    #set the SEP indices for each sentence token to zero
    attention_masks_without_special_tok[row_idx, col_idx] = 0

    return input_ids, attention_masks, attention_masks_without_special_tok

#initialize model
#output_hidden_states = True will give us all hiddenn states for all layers
# pretrained_model = RobertaModel.from_pretrained(PRETRAINED_MODEL ,output_hidden_states = True)
pretrained_model = BertModel.from_pretrained(PRETRAINED_MODEL ,output_hidden_states = True)

#put this in eval mode so since we do not plan to do backprop
pretrained_model.eval()

### Lets pick the sentences that we would run through and visualize distance/similarity
#List of tupes :
#(sentence, label_id)
# label_id == 0 == negative
# label_id == 1 == positive

# xyx
# sents_and_labs = [
#          ('This taffy is so good.  It is very soft and chewy.  The flavors are amazing.  I would definitely recommend you buying it.  Very satisfying!!', 1),
#         # ('This is a good film. This is very funny. Yet after this film there were no good Ernest films!', 1),
#          ('Just love the interplay between two great characters of stage & screen - Veidt & Barrymore', 1),
#          ('Hated it with all my being. Worst movie ever. Mentally- scarred. Help me. It was that bad.TRUST ME!!!', 0),
#          ("This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.", 0)
# ]
sents_and_labs = [
         ('什么牌子的轮胎好', 1),
        # ('This is a good film. This is very funny. Yet after this film there were no good Ernest films!', 1),
         ('我的是什么牌子的轮胎', 1),
         ('华为手机好还是联想手机好woma', 0),
         ("联想手机好还是华为手机好？wodema", 0)
]


sents = [s for s,l in sents_and_labs]
print(sents)

def get_preds(sentences, tokenizer_obj, model_obj):
  '''
  Quick function to extract hidden states and masks from the sentences and model passed
  '''
  #Run the sentences through tokenizer
  input_ids, att_msks, attention_masks_wo_special_tok = preprocessing_for_bert(sentences, tokenizer_obj)
  #Run the sentences through the model
  outputs = model_obj(input_ids, att_msks)

  #Lengths of each sentence
  sent_lens = att_msks.sum(1).tolist()

  #calculate unique vocab
  # #get the tokenized version of each sentence (text form, to label things in the plot)
  tokenized_sents = [tokenizer_obj.convert_ids_to_tokens(i) for i in input_ids]
  return {
      'hidden_states':outputs[2],
      'pooled_output': outputs[1],
      'attention_masks': att_msks,
      'attention_masks_without_special_tok': attention_masks_wo_special_tok,
      'tokenized_sents': tokenized_sents,
      'sentences': sentences,
      'sent_lengths': sent_lens
  }

pretrained_preds = get_preds(sents, tokenizer, pretrained_model)

def plt_dists(dists, sentences_and_labels, dims=2,  title="", xrange=[-.5,.5], yrange=[-.5,.5], zrange=[-0.5, 0.5]):
    '''
    Plot distances using MDS in 2D/3D 
    dists: precomputed distance matrix
    sentences_and_labels: tuples of sentence and label_ids
    dims: 2/3 for 2 or 3 dimensional plot, defaults to 2 for any other value passed
    words_of_interest: list of words to highlight with a different color
    title: title for the plot
    '''
    #get the sentence text and labels to pass to the plot
    sents, color = zip(*sentences_and_labels)

    #https://community.plotly.com/t/plotly-colours-list/11730/6
    colorscale = [[0, 'deeppink'], [1, 'yellow']] #, [2, 'greens'], [3, 'reds'], [4, 'blues']]

    #dists is precomputed using cosine similarity/other other metric and passed
    #calculate MDS with number of dims passed
    mds = manifold.MDS(n_components=dims, dissimilarity="precomputed", random_state=60, max_iter=90000)
    results = mds.fit(dists)

    #get coodinates for each point
    coords = results.embedding_

    #plot 3d/2d
    if dims == 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers+text', textposition="top center", text=sents,
            marker=dict(size=12, color=color, colorscale=colorscale, opacity=0.8)
        )])
    else:
        fig = go.Figure(data=[go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            text=sents, textfont=dict(family="SimHei, FangSong, Times New Roman"), textposition="top center", mode='markers+text',
            marker=dict(size=12,color=color,colorscale=colorscale, opacity=0.8)
        )])

    fig.update_layout(template="plotly_dark")
    if title!="":
        fig.update_layout(title_text=title)
        fig.update_layout(titlefont=dict(family='Times New Roman, monospace',
                        size=14, color='cornflowerblue'))

    #update the axes ranges
    fig.update_layout(yaxis=dict(range=yrange))
    fig.update_layout(xaxis=dict(range=xrange))
    fig.update_traces(textfont_size=10)

    #TO DO: fix this. I could not get this to work. somehow the library does not like the zaxis. 
    # if dims==3:
    # fig.update_layout(zaxis=dict(range=zrange))
    # VSCode Server Error: connect ENOENT /run/user/0/vscode-ipc-1b9a619e-93e6-42b1-9741-dfbd419fa370.sock
    # fig.show() 

    # plotly.io.orca.config.executable = '/home/XXX/anaconda3/bin/orca' 缺少动态链接库

    path = 'fig2.jpg'
    # fig.save(path)
    fig.write_image(path, engine="kaleido")

def get_word_vectors(hidden_layers_form_arch, token_index=None, mode='average', top_n_layers=4):
    '''
    retrieve vectors for all tokens from the top n layers and return a concatenated, averaged or summed vector 
    hidden_layers_form_arch: tuple returned by the transformer library
    token_index: None/Index:
    If None: Returns all the tokens 
    If Index: Returns vectors for that index in each sentence 

    mode=
        'average' : avg last n layers
        'concat': concatenate last n layers
        'sum' : sum last n layers
        'last': return embeddings only from last layer
        'second_last': return embeddings only from second last layer

    top_n_layers: number of top layers to concatenate/ average / sum
    '''

    vecs = None
    if mode == 'concat':
        vecs = torch.cat(hidden_layers_form_arch[-top_n_layers:], dim=2)

    if mode == 'average':
        vecs = torch.stack(hidden_layers_form_arch[-top_n_layers:]).mean(0)

    if mode == 'sum':
        vecs = torch.stack(hidden_layers_form_arch[-top_n_layers:]).sum(0)

    if mode == 'last':
        vecs = hidden_layers_form_arch[-1:][0]

    if mode == 'second_last':
        vecs = hidden_layers_form_arch[-2:-1][0]

    if vecs is not None and token_index:
        #if a token index is passed, return values for a particular index in the sequence instead of vectors for all
        return vecs.permute(1,0,2)[token_index]
    return vecs

def get_sent_vectors(input_states, att_mask):
    '''
    get a sentence vector by averaging over all word vectors -> this could come from any layers or averaged themselves (see get_all_token_vectors function)
    input_states: [batch_size x seq_len x vector_dims] -> e.g. output from  hidden stats from a particular layer
    att_mask: attention mask passed should have already maseked the special tokens too i.e. CLS/SEP/<s>/special tokens masked out with 0 -> [batch_size x max_seq_length]
    ref: https://stackoverflow.com/questions/61956893/how-to-mask-a-3d-tensor-with-2d-mask-and-keep-the-dimensions-of-original-vector
    '''

    # print(input_states.shape) #-> [batch_size x seq_len x vector_dim]

    #Let's get sentence lengths for each sentence
    sent_lengths = att_mask.sum(1)      #att_mask has a 1 against each valid token and 0 otherwise 

    #create a new 3rd dim and broadcast the attention mask across it -> this will allow us to use this mask with the 3d tensor input_hidden_states
    att_mask_ = att_mask.unsqueeze(-1).expand(input_states.size()) 

    #use mask to 0 out all the values against special tokens like CLS, SEP , <s> using mask
    masked_states = input_states*att_mask_

    #calculate average 
    sums = masked_states.sum(1) 
    avg = sums/sent_lengths[:, None]
    return avg

def eval_vectors(model_output, sentences_and_labels, wrd_vec_mode='concat', 
                 wrd_vec_top_n_layers=4, viz_dims=2, 
                 sentence_emb_mode='average_word_vectors',
                 title_prefix=None,
                 plt_xrange=[-0.05, 0.05], plt_yrange=[-0.05, 0.05], plt_zrange=[-0.05, 0.05]):
    '''
    Get vectors for all sentences and visualize them based on cosine distance between them

    model_output: model output extracted as a dictionary from get_preds function
    sentences_and_labels: tuple of sentence and labels_ids
    att_msk: attention mask that also marks the special tokens (CLS/SEP etc.) as 0
    mode=
        'average' : avg last n layers
        'concat': concatenate last n layers
        'sum' : sum last n layers
        'last': return embeddings only from last layer
        'second_last': return embeddings only from second last layer
    viz_dims:2/3 for 2D/3D plot
    title_prefix: String to add before the descriptive title. Can be used to add model name etc.
    '''
    title_wrd_emv = "{} across {} layers".format(wrd_vec_mode, wrd_vec_top_n_layers)

    #get word vectors for all words in the sentence
    if sentence_emb_mode == 'average_word_vectors':
        title_sent_emb = "average(word vectors in the sentence); Sentence Distance: Cosine"
        word_vecs_across_sent = get_word_vectors(model_output['hidden_states'], mode=wrd_vec_mode, token_index=None, top_n_layers=wrd_vec_top_n_layers)    #returns [batch_size x seq_len x vector_dim]
        sent_vecs = get_sent_vectors(word_vecs_across_sent, model_output['attention_masks_without_special_tok'])
    else:
        title_sent_emb = "First tok (CLS) vector; Sentence Distance: Cosine"
        #Get the pooled output from the first token (e.g. CLS token in case of BERT)

        #Note from https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        #This output is usually not a good summary of the semantic content of the 
        #input, you’re often better with averaging or 
        #pooling the sequence of hidden-states for the whole input sequence.
        print("inside")
        sent_vecs =  model_output['pooled_output'] #vector 

    if title_prefix:
        final_title = '{} Word Vec: {}; Sentence Vector: {}'.format(title_prefix, title_wrd_emv, title_sent_emb)
    else:
        final_title = 'Word Vec: {}; Sentence Vector: {}'.format(title_wrd_emv, title_sent_emb)
    mat = sent_vecs.detach().numpy()
    plt_dists(cosine_distances(mat), sentences_and_labels=sentences_and_labels, dims=viz_dims, title=final_title, xrange=plt_xrange, yrange=plt_yrange, zrange=plt_zrange)

# eval_vectors(pretrained_preds, sents_and_labs, wrd_vec_mode='concat', sentence_emb_mode="average_word_vectors", 
#              plt_xrange=[-0.03, 0.03], plt_yrange=[-0.03, 0.03], title_prefix="Pretrained model:")

# eval_vectors(pretrained_preds, sents_and_labs, wrd_vec_mode='concat', sentence_emb_mode="pooled_output", 
#              plt_xrange=[-0.03, 0.03], plt_yrange=[-0.03, 0.03], title_prefix="Pretrained model:")

# eval_vectors(pretrained_preds, sents_and_labs, wrd_vec_mode='average', sentence_emb_mode="average_word_vectors", title_prefix="Pretrained model:",
#              plt_xrange=[-0.03, 0.03], plt_yrange=[-0.03, 0.03])

# eval_vectors(pretrained_preds, sents_and_labs, wrd_vec_mode='second_last', sentence_emb_mode="average_word_vectors", 
#              plt_xrange=[-0.03, 0.04], plt_yrange=[-0.03, 0.03], title_prefix="Pretrained model:")


# WMD 
def get_vector_for_each_token_position(hidden_layers_form_arch, token_index=0, mode='average', top_n_layers=4):
    '''
    retrieve vectors for a token_index from the top n layers and return a concatenated, averaged or summed vector 
    hidden_layers_form_arch: tuple returned by the transformer library
    token_index: index of the token for which a vector is desired
    mode=
        'average' : avg last n layers
        'concat': concatenate last n layers
        'sum' : sum last n layers
        'last': return embeddings only from last layer
        'second_last': return embeddings only from second last layer

    top_n_layers: number of top layers to concatenate/ average / sum
    '''
    if mode == 'concat':
        #concatenate last 4 layer outputs -> returns [batch_size x seq_len x dim]
        #permute(1,0,2) swaps the the batch and seq_len dim , making it easy to return all the vectors for a particular token position
        return torch.cat(hidden_layers_form_arch[-top_n_layers:], dim=2).permute(1,0,2)[token_index]

    if mode == 'average':
        #avg last 4 layer outputs -> returns [batch_size x seq_len x dim]
        return torch.stack(hidden_layers_form_arch[-top_n_layers:]).mean(0).permute(1,0,2)[token_index]


    if mode == 'sum':
        #sum last 4 layer outputs -> returns [batch_size x seq_len x dim]
        return torch.stack(hidden_layers_form_arch[-top_n_layers:]).sum(0).permute(1,0,2)[token_index]


    if mode == 'last':
        #last layer output -> returns [batch_size x seq_len x dim]
        return hidden_layers_form_arch[-1:][0].permute(1,0,2)[token_index]

    if mode == 'second_last':
        #last layer output -> returns [batch_size x seq_len x dim]
        return hidden_layers_form_arch[-2:-1][0].permute(1,0,2)[token_index]

    return None

def build_word_embedding_lookup(model_output, wrd_vec_mode='concat',top_n_layers=4, max_len=MAX_LEN):
    '''
    build a embedding lookup - this will be needed when we do need to pull up vectors for any word while calculating wmd
    model_output: model output extracted as a dictionary from get_preds function; should include 'hidden_states', 'tokenized_sents', 'sent_lengths'
    wrd_vec_mode: concat/average/sum/last/second_last - way to extract word embeddings from the architecture
    top_n_layers: number of layers to work on to get word vectors using the wrd_vec_mode
    max_len: max length of the sentence for the architecture
    returns:
    vecs: a dict with keys as tokens and sentence number (e.g. date in sent 0 becomes date_0), and values as vectors extracted from bert like models
    documents: dictionary with sentence number as key and tokens like date_0 joined with a space as a string
    '''
    vecs = dict()
    documents = dict()

    for token_ind in range(max_len):
        if token_ind == 0:
            #ignore CLS
            continue

        vectors = get_vector_for_each_token_position(model_output['hidden_states'], token_index=token_ind, mode=wrd_vec_mode, top_n_layers=top_n_layers)
        # print(vectors)
        for sent_ind, sent_len in enumerate(model_output['sent_lengths']):
            if token_ind < sent_len-1:  #ignore SEP which will be at sent_len-1 index
                txt = model_output['tokenized_sents'][sent_ind][token_ind]+"_"+str(sent_ind)
                # print(txt)
                #store the token and its vector -> this will be our lookup storage for vectors
                vecs[txt] = vectors[sent_ind].detach().numpy()
                # print(vecs[txt])
                #store this so that we can do comparisons
                if sent_ind not in documents:
                    documents[sent_ind] = txt
                else:
                    documents[sent_ind] += ' ' + txt 
    return vecs, documents
    # vecs 记录[char_sentenceIndex]的词向量
    # documents 记录[第某个句子的序列]的char_sentenceIndex

#Modified from https://github.com/src-d/wmd-relax/blob/master/wmd/__init__.py
#class to extract and calculate word movers distance using bert 
class SimilarityWMD(object):
    def __init__(self, embedding_dict, sklearn_euclidean_distances=True, **kwargs):
        """
        :param embedding_dict: a dictionary to look up vectors 
        :param only_alpha: Indicates whether only alpha tokens must be used.
        :param frequency_processor: The function which is applied to raw \
                                    token frequencies.

        :type frequency_processor: callable
        """

        self.frequency_processor = kwargs.get(
            "frequency_processor", lambda t, f: np.log(1 + f))
        self.embedding_dict = embedding_dict
        # print(type(self.embedding_dict))
        # print(self.embedding_dict)
        #get embed size 
        self.emb_size = self.embedding_dict[next(iter(self.embedding_dict))].shape[0] # 3072
        # while True:
        #     try:
        #         self.emb_size = self.embedding_dict[next(iter(self.embedding_dict))].shape[0]
        #     except StopIteration:
        #         #遇到StopIteration就退出循环
        #         break
        self.sklearn_euclidean_distances = sklearn_euclidean_distances

    def _get_normalized_item(self, item):
        '''
        get id and find a vector for the corresponding id in the embedding lookup
        '''
        v = self.embedding_dict[item]
        return v/v.sum()

    def _dist_fn(self, u, v):
      return libwmdrelax.emd(u, v, self.dists)
    
    def _calc_euclidean_distances(self, evec):
        if self.sklearn_euclidean_distances:
            #call sklearn.metrics.pairwise.euclidean_distances
            return euclidean_distances(evec)

        evec_sqr = (evec * evec).sum(axis=1)
        dists = evec_sqr - 2 * evec.dot(evec.T) + evec_sqr[:, np.newaxis]
        dists[dists < 0] = 0
        dists = np.sqrt(dists)
        for i in range(len(dists)):
                dists[i, i] = 0
        return dists

    def compute_similarity(self, docs, target=0):
        """
        Calculates the similarity between two spaCy documents. Extracts the
        nBOW from them and evaluates the WMD.
        :return: The calculated similarity.
        :rtype: float.
        """
        
        #{'word1': 0.6931471805599453,...}
        #generates word -> freq mapping for each doc
        docs_nbow = [self._convert_document(d) for d in docs]

        #get vocab with indices for each 
        #{239326000841: 0, 286393583696: 1, ...}
        vocabulary = set()
        for distribution in docs_nbow:
            vocabulary = vocabulary.union(set(distribution)) 

        vocabulary = {w: i for i, w in enumerate(sorted(vocabulary))}

        '''
        #generate nbow
        e.g.
        [0.14285715 0.14285715 0.         0.         0.         0.
        0.14285715 0.14285715 0.         0.14285715 0.         0.14285715
        0.         0.14285715]
        '''
        weights = list()
        for d in docs_nbow:
          weights.append(self._generate_weights(d, vocabulary))

        # 获得词频作为di, dj

        evec = np.zeros((len(vocabulary), self.emb_size), dtype=np.float32)
        
        for w, i in vocabulary.items():
            evec[i] = self._get_normalized_item(w)

        
        #calculate euclidean_distances between all pairs of vectors 
        self.dists = self._calc_euclidean_distances(evec)
        
        print(weights)
        #calculate word movers distance for all our sentences
        wmd_dists = pdist(weights, self._dist_fn)
        # 计算了所有对的距离
        print(type(wmd_dists))
        print(wmd_dists)

        # 计算target与其他的距离
        target_wmd_distance = [self._dist_fn(weights[target], weights[i]) for i in range(len(weights)) if i != target ]

        #return a datafrrame NxN (N = number of sentences) with distances between each pair
        #return pd.DataFrame(squareform(wmd_dists), index=docs, columns=docs)
        return squareform(wmd_dists), target_wmd_distance

    def _convert_document(self, doc):
        wrds = defaultdict(int)
        for t in doc.split():
            wrds[t] += 1
        return {t: self.frequency_processor(t, v) for t, v in wrds.items()}

    def _generate_weights(self, doc, vocabulary):
        w = np.zeros(len(vocabulary), dtype=np.float32)
        for t, v in doc.items():
            w[vocabulary[t]] = v
        w /= w.sum()
        return w

def eval_using_wmd(model_output, sentences_and_labels, wrd_vec_mode='concat', 
                   viz_dims=2, wrd_vec_top_n_layers=4, 
                   title_prefix=None,
                   plt_xrange=[-0.03, 0.03], plt_yrange=[-0.03, 0.03], plt_zrange=[-0.05, 0.05]):
    '''
    model_output: model output extracted as a dictionary from get_preds function
    sentences_and_labels: tuple of sentence and labels_ids
    wrd_vec_top_n_layers: number of layers to use while extracting word embeddings
    wrd_vec_mode=
            'average' : avg last n layers
            'concat': concatenate last n layers
            'sum' : sum last n layers
            'last': return embeddings only from last layer
            'second_last': return embeddings only from second last layer
    viz_dims:2/3 for 2D/3D plot
    title_prefix: String to add before the descriptive title. Can be used to add model name etc.
    '''
    #get all vectors for all words in each sentence
    vecs , documents = build_word_embedding_lookup(model_output, wrd_vec_mode=wrd_vec_mode, top_n_layers=wrd_vec_top_n_layers)
    
    #calculate the word movers distance
    print(vecs)
    dist_matrix, target_wmd_distance = SimilarityWMD(vecs).compute_similarity([documents[i] for i in range(len(documents))])
    print(type(dist_matrix))

    print(target_wmd_distance)
    title_wrd_emv = "{} across {} layers".format(wrd_vec_mode, wrd_vec_top_n_layers)

    if title_prefix:
        final_title = '{} Word Vec: {}; Sentence Distance: Word Movers Distance'.format(title_prefix, title_wrd_emv)
    else:
        final_title = 'Word Vec: {}; Sentence Distance: Word Movers Distance'.format(title_wrd_emv)

    #plot distances
    plt_dists(dist_matrix, sentences_and_labels=sentences_and_labels, dims=viz_dims, title=final_title, xrange=plt_xrange, yrange=plt_yrange, zrange=plt_zrange)

eval_using_wmd(pretrained_preds, sents_and_labs, wrd_vec_mode='concat', plt_xrange=[-0.4, 0.4], plt_yrange=[-0.4, 0.4], title_prefix="Pretrained model:")

# eval_using_wmd(pretrained_preds, sents_and_labs, wrd_vec_mode='average', plt_xrange=[-0.4, 0.4], plt_yrange=[-0.4, 0.4], title_prefix="Pretrained model:")

# eval_using_wmd(pretrained_preds, sents_and_labs, wrd_vec_mode='second_last', plt_xrange=[-0.4, 0.4], plt_yrange=[-0.4, 0.4], title_prefix="Pretrained model:")

# eval_using_wmd(pretrained_preds, sents_and_labs, wrd_vec_mode='last', plt_xrange=[-0.4, 0.4], plt_yrange=[-0.4, 0.4], title_prefix="Pretrained model:")
