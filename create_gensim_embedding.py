from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


glove_input_file = '../glove.6B/glove.6B.300d.txt'
word2vec_output_file = '../glove.6B/word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
model.save('glove.300d.gensim')
