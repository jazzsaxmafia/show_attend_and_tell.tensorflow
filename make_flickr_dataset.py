import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *

vgg_model = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

annotation_path = './data/results_20130124.token'
flickr_image_path = '../show_attend_and_tell/images/flickr30k-images/'
feat_path = './data/feats.npy'
annotation_result_path = './data/annotations.pickle'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

unique_images = annotations['image'].unique()
image_df = pd.DataFrame({'image':unique_images, 'image_id':range(len(unique_images))})

annotations = pd.merge(annotations, image_df)
annotations.to_pickle(annotation_result_path)

if not os.path.exists(feat_path):
    feats = cnn.get_features(unique_images, layers='conv5_3', layer_sizes=[512,14,14])
    np.save(feat_path, feats)

