import numpy as np
import copy
from .encoders import DenseEncoder, SparseEncoder
import tensorflow as tf
import tensorflow_hub as hub
import official.nlp.bert.tokenization as tokenization

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True,name='bert_layer')
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=25):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0]*pad_len
        pad_masks = [1] * len(input_sequence) + [0]*pad_len
        segment_ids = [0]*max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def find_min_idx(array):
  """
  Returns: the index of min in array
  """
  min_idx = 0

  for idx, item in enumerate(array):
    if item < array[min_idx]:
      min_idx = idx

  return min_idx 

def find_topk_candidate(array, topk):
  topk_idx = []

  for idx,item in enumerate(array):
    if len(topk_idx) < topk:
      topk_idx.append(idx)
    else:
      min_idx = find_min_idx(array[topk_idx])
      if item > array[topk_idx[min_idx]]:
        topk_idx[min_idx] = idx
  return np.array(topk_idx)

def sort_candidate_descening(array_value, array_idx):

  for i in range(len(array_value)-1):
    max_idx = i
    for j in range(i+1, len(array_value)):
      if array_value[j] > array_value[max_idx]:
        max_idx = j
    if max_idx != i:
      
      tmp = array_value[i]
      array_value[i] = array_value[max_idx]
      array_value[max_idx] = tmp

      tmp = array_idx[i]
      array_idx[i] = array_idx[max_idx]
      array_idx[max_idx] = tmp
  return array_idx

  # define function to get sparse candidates and dense candidates indices of query

def get_sparse_candidate_indices(query_text, corpus, sparse_encoder, topk=20):
  """
  Parameters:
    query_text: list of query string
    corpus: list of text in dictionary
    sparse_encoder: sparse encoder which embeds text to vector base tf-idf method
  
  Returns: 
    np.darray: matrix of indices of candidate in dictionary base on sparse score. Shape: (len(query), topk)
  """
  sparse_encoder.fit(corpus)
  query_sparse_embeddings = sparse_encoder.transform(query_text)
  dictionary_sparse_embeddings = sparse_encoder.transform(corpus)

  score_sparse_matrix = np.dot(query_sparse_embeddings, dictionary_sparse_embeddings.transpose())
  
  sparse_candidates = []
  for i in range(0, len(score_sparse_matrix)):
    topk_candidates_sparse =  np.argpartition(score_sparse_matrix[i], -topk)[-topk:]  # get n_sparse candidate first
    array_value = copy.deepcopy(score_sparse_matrix[i][topk_candidates_sparse])
    topk_candidates_sparse = sort_candidate_descening(array_value, topk_candidates_sparse)
    sparse_candidates.append(topk_candidates_sparse)

  sparse_candidates = np.array(sparse_candidates)
  return sparse_candidates

def get_dense_candidate_indices(query_text, corpus, bert_layer,topk=20):
  """
  Parameters:
    query_text: list of query string
    corpus: list of text in dictionary
    bert_layer: bert-pre-trained
  
  Returns: 
    np.darray: matrix of indices of candidate in dictionary base on dense score. Shape: (len(query), topk)
  """
  query_tokens = bert_encode(query_text, tokenizer, max_len=25)
  dictionary_tokens = bert_encode(corpus, tokenizer, max_len=25)
  dense_encoder = DenseEncoder(bert_layer, max_len=25)
  query_dense_embeddings = dense_encoder.get_dense_embedding(query_tokens) # [None, 768]
  dictionary_dense_embeddings =dense_encoder.get_dense_embedding(dictionary_tokens) # [None, 768]
  
  query_dense_score = np.dot(query_dense_embeddings, tf.transpose(dictionary_dense_embeddings, perm=[1,0]))
  
  candidates_dense = []
  for i in range(0, len(query_dense_score)):
    topk_candidates_dense = np.argpartition(query_dense_score[i], -topk)[-topk:]  # get n_sparse candidate first
    array_value = copy.deepcopy(query_dense_score[i][topk_candidates_dense])
    topk_candidates_dense = sort_candidate_descening(array_value, topk_candidates_dense)
    candidates_dense.append(topk_candidates_dense)

  candidates_dense = np.array(candidates_dense)
  return candidates_dense

def get_query_candidates_indices(query_text, corpus, sparse_encoder, bert_layer, topk=20):
  """
  Parameters:
    candidate_dense_indices: matrix of dense candidate indices
    candidate_sparse_indices: matrix of sparse candidate indices
    topk: number of candidates of a query
  
  Returns: 
    np.darray: matrix of indices of candidate in dictionary base on dense and sparse score. Shape: (len(query), topk)
  """

  sparse_candidate_indices = get_sparse_candidate_indices(query_text, corpus, sparse_encoder,topk)
  dense_candidate_indices = get_dense_candidate_indices(query_text, corpus, bert_layer, topk=30)

  candidates_indices = np.empty((sparse_candidate_indices.shape[0],topk))
  candidates_indices[:,0:11] = sparse_candidate_indices[:,0:11]
  for idx, row in enumerate(candidates_indices):
    n = 10
    for i in range(topk):
      if dense_candidate_indices[idx][i] not in row:
        row[n] = dense_candidate_indices[idx][i] 
        n += 1;      
      if n == topk: 
        break;
    
  return candidates_indices

def get_dense_candidate_embeddings(candidate_indices, dictionary, bert_layer):
  """
  Parameters:
    candidate_indices: matrix of candidate indices
    dictionary: dictionary
    bert_layer: bert-pretrained packaged in layer
  
  Returns: 
    np.darray: matrix of dense embeddings of candidates of queries. Shape: (len(query), topk, 768)
  """

  candidates_raw_text = []
  for row in candidate_indices:
    row_texts = dictionary.data[row.astype(int).tolist(),0]
    candidates_raw_text.append(row_texts)

  candidates_raw_text = np.array(candidates_raw_text)
  candidate_tokens = []
  for row in candidates_raw_text:
    row_tokens = bert_encode(row, tokenizer,max_len=25)
    candidate_tokens.append(row_tokens)
  
  dense_encoder = DenseEncoder(bert_layer, max_len=25)
  candidate_embeddings = []
  for row in candidate_tokens:
    row_embeddings = dense_encoder.get_dense_embedding(row)
    candidate_embeddings.append(row_embeddings)

  candidate_embeddings = np.array(candidate_embeddings)
  return candidate_embeddings

#candidate_embeddings = get_query_candidates_embeddings(candidates_indices, training_dictionary, bert_layer)
def get_sparse_candidate_score(candidates_indices, sparse_score_matix):
  """
  Parameters:
    candidate_indices: matrix of candidate indices
    sparse_score_matix: sparse score of all queries
    bert_layer: bert-pretrained packaged in layer
  
  Returns: 
    np.darray: matrix of dense embeddings of candidates of queries. Shape: (len(query), topk, 768)
  """

  candidate_sparse_score = []
  for idx,row in enumerate(candidates_indices):
    row_sparse_score = sparse_score_matix[idx][row.astype(int).tolist()]
    candidate_sparse_score.append(row_sparse_score)
  
  candidate_sparse_score = np.array(candidate_sparse_score)

  return candidate_sparse_score

def get_labels_of_candidates(true_labels, candidates_indices, dictionary):
  label_candidates = []
  for idx, row in enumerate(candidates_indices):

    row_label = []
    predict_label = dictionary.data[row.astype(int).tolist(),1]
    for i, labels in enumerate(predict_label):
      row_label.append(0)
      labels = labels.split('|')
      for label in labels:
        if label in true_labels[idx]:
          row_label[i] = 1
    label_candidates.append(row_label)

  return label_candidates

def get_sparse_query_score(sparse_encoder, corpus, query_text):

  sparse_encoder.fit(corpus)
  query_sparse_embeddings = sparse_encoder.transform(query_text)
  dictionary_sparse_embeddings = sparse_encoder.transform(corpus)
  score_sparse_matrix = np.dot(query_sparse_embeddings, dictionary_sparse_embeddings.transpose())

  return score_sparse_matrix


