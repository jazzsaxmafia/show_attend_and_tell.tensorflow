from taeksoo.cnn_util import *
import pandas as pd
import numpy as np
import os
import scipy
import ipdb
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer

annotation_path = '/home/taeksoo/Study/show_attend_and_tell/data/flickr30k/results_20130124.token'
flickr_image_path = './images/flickr30k/'

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

captions = annotations['caption'].values

vectorizer = CountVectorizer(max_features=9995, token_pattern='\\b\\w+\\b').fit(captions)
dictionary = vectorizer.vocabulary_
dictionary = pd.Series(dictionary) + 5

#dictionary = dictionary_series.to_dict()
dictionary['<eos>'] = 0
dictionary['UNK'] = 1
dictionary[','] = 2
dictionary['!'] = 3
dictionary['?'] = 4

with open('/home/taeksoo/Study/show_attend_and_tell/data/flickr30k/dictionary.pkl', 'wb') as f:
    cPickle.dump(dictionary, f)
with open('/home/taeksoo/Study/show_attend_and_tell/data/flickr30k/vectorizer.pkl', 'wb') as f:
    cPickle.dump(vectorizer, f)
