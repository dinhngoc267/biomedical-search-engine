import keras
import tensorflow as tf

import keras.backend as K
from biosyn.dataset import QueryDataset, DictionaryDataset
import numpy as np
from biosyn.AdamLRM import AdamLRM
from biosyn.encoders import SparseEncoder, DenseEncoder
from biosyn.utils import *

# load data
training_query = QueryDataset('./biosyn/dataset/TAC2017/train')
training_dictionary = DictionaryDataset('./biosyn/dataset/dictionary/train_dictionary.txt')
test_query = QueryDataset('./biosyn/dataset/TAC2017/test')
test_dictionary = DictionaryDataset('./biosyn/dataset/dictionary/test_dictionary.txt')


class Scalar(tf.keras.layers.Layer):
  def __init__(self,name):
    super(Scalar, self).__init__()
    self._name = name
  def build(self,input_shape) :
    self.W = K.variable(0)
    self._trainable_weights=[self.W]
    super().build(input_shape)
  def call(self,inputs):
    return self.W*inputs

# Define model ReRanker
def marginal_loss(output, target):
  predict = tf.nn.softmax(tf.cast(output, dtype=tf.float32))
  loss = predict*target
  loss = K.sum(loss,axis=-1)                  # sum all positive scores

  loss = loss[loss > 0]                     # filter sets with at least one positives
  #loss = K.clip(loss, min_value=1e-9, max_value=1) # for numerical stability
  loss = -K.log(loss)                   # for negative log likelihood
  if len(loss) == 0:
      loss = K.sum(loss)                     # will return zero loss
  else:
      loss = K.mean(loss)
  return loss

class BioSyn():
  def __init__(self, bert_layer, pretrained_weight, dictionary, topk=20, max_len=25):
    self.bert_layer = bert_layer
    self.topk = topk
    self.dictionary = dictionary

    query_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="query_word_ids")
    query_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="query_mask")
    query_segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="query_segment_ids")

    pooled_output, sequence_output = self.bert_layer([query_word_ids, query_mask, query_segment_ids])
    query_dense_embeddings = sequence_output[:, 0, :] # [None, 768]

    # score: 
    candidates_dense_embeddings = tf.keras.Input(shape=(topk,768,), dtype=tf.float32, name="candidates_dense_embeddings")
    candidates_dense_score = keras.layers.Dot(axes=(2,1),name="dense_score")([candidates_dense_embeddings,query_dense_embeddings])
    
    batch_size = candidates_dense_score.shape[0] 
    candidate_sparse_score = tf.keras.Input(shape=(batch_size,), dtype=tf.float32, name="candidate_sparse_score")
    scaling_sparse_score = Scalar(name='sparse_weight')(candidate_sparse_score)

    score = scaling_sparse_score + candidates_dense_score

    lr_multiplier = {
      'bert_layer':1, # optimize with a smaller learning rate
      'sparse_weight':0.5e+3   # optimize  with a larger learning rate
      }
    opt = AdamLRM(learning_rate=1e-5, lr_multiplier=lr_multiplier)

    model =  tf.keras.models.Model(inputs = [[query_word_ids, query_mask, query_segment_ids],candidates_dense_embeddings, candidate_sparse_score], outputs=score )
    model.compile(optimizer=opt,loss=marginal_loss)
    model.load_weights(pretrained_weight)
    sparse_weight = model.get_layer(name='sparse_weight').get_weights()[0]
    self.sparse_weight = sparse_weight

    #prepare dictionary vector
    #sparse vector
    self.sparse_encoder = SparseEncoder()
    self.sparse_encoder.fit(self.dictionary.data[:,0])

    dictionary_sparse_embeddings = self.sparse_encoder.transform(self.dictionary.data[:,0])
    self.dictionary_sparse_embeddings = dictionary_sparse_embeddings
    #dense vector
    dictionary_tokens = bert_encode(self.dictionary.data[:,0], tokenizer, max_len)
    self.dense_encoder = DenseEncoder(self.bert_layer, max_len)
    dictionary_dense_embeddings = self.dense_encoder.get_dense_embedding(dictionary_tokens) 
    self.dictionary_dense_embeddings = dictionary_dense_embeddings

  def retreival(self, query, max_len=25, topk=20):
    query_sparse_embeddings = self.sparse_encoder.transform(query)
    query_sparse_score = np.dot(query_sparse_embeddings, self.dictionary_sparse_embeddings.transpose())

    query_tokens = bert_encode(query, tokenizer, max_len)
    query_dense_embeddings = self.dense_encoder.get_dense_embedding(query_tokens) 
    query_dense_score = np.dot(query_dense_embeddings, tf.transpose(self.dictionary_dense_embeddings, perm=[1,0]))  
  
    query_score = query_sparse_score*self.sparse_weight + query_dense_score

    candidates_indices = []
    for i in range(0, len(query_score)):
      topk_candidates_dense = np.argpartition(query_score[i], -topk)[-topk:]  # get n_sparse candidate first
      array_value = copy.deepcopy(query_score[i][topk_candidates_dense])
      topk_candidates_dense = sort_candidate_descening(array_value, topk_candidates_dense)
      candidates_indices.append(topk_candidates_dense)

    candidates_indices = np.array(candidates_indices)
    return candidates_indices

  def evaluate(true_labels,candidate_indices, dictionary):
    get_labels_of_candidates
    label_candidates = get_labels_of_candidates(true_labels,candidate_indices, dictionary)
    n = len(label_candidates)
    top_1 = top_5 = top_10 = top_20 = 0
    for row in label_candidates:
      if row[0] == 1:
        top_1 += 1
      if 1 in row[0:5]:
        top_5 += 1
      if 1 in row[0:10]:
        top_10 += 1
      if 1 in row[0:20]:
        top_20 += 1
      
    return [top_1/n, top_5/n, top_10/n, top_20/n]
#candidates_test_indices = retreival(training_query, training_dictionary, bert_layer, sparse_weight[0])
model = BioSyn(bert_layer, './biosyn/biosyn_tac2017_epoch_5.h5',test_dictionary)
#candidates_test_indices = model.retreival(test_query.data[:,0])
#res = model.evaluate(test_query.data[:,1],candidates_test_indices, test_dictionary)
#print('Top 1: {:.2f}% '.format(res[0]*100) + 'Top 5: {:.2f}% '.format(res[1]*100) + 'Top 10: {:.2f}% '.format(res[2]*100) + 'Top 20: {:.2f}% '.format(res[3]*100))