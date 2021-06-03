import tensorflow as tf 
from sklearn.feature_extraction.text import TfidfVectorizer 


class DenseEncoder():
  def __init__(self, bert_layer, max_len=25):
    
    input_word_ids = tf.keras.Input(shape=(max_len,),dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,),dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask,segment_ids])
    #output = bert_layer([input_word_ids, input_mask,segment_ids])[0]
    clf_output = sequence_output[:, 0, :]

    self.model = tf.keras.models.Model(inputs=[input_word_ids,input_mask,segment_ids], outputs=clf_output)

  def get_model(self):
    return self.model

  def get_dense_embedding(self,input):
    return self.model.predict([input],batch_size=1024)


class SparseEncoder(object):
  def __init__(self):
    self.encoder = TfidfVectorizer(analyzer='char', ngram_range=(1,2))

  def fit(self, train_corpus):
    self.encoder.fit(train_corpus)

    return self
  
  def transform(self, mentions):
    vec = self.encoder.transform(mentions).toarray()
    #vec = tf.constant(vec, dtype=tf.float32)
    return vec
  
  def __call__(self, mentions):
    return self.transform(mentions)
