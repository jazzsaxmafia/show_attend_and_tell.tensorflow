# -*- coding: utf-8 -*-
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import pandas as pd
import numpy as np
import os
import theano
import theano.tensor as T
import ipdb
import cPickle

from keras.preprocessing import sequence
from keras import activations, initializations
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.utils.theano_utils import shared_scalar, shared_zeros, sharedX, alloc_zeros_matrix

from theano import config
from taeksoo.cnn_util import *

trng = RandomStreams(1234)
def dropout(X):
    if train:
        X *= trng.binomial(X.shape, p=0.5, dtype=theano.config.floatX)
        X /= 0.5

    return X

def ortho_weight(ndim):

    W = np.random.randn(ndim, ndim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')


############# Building Models ################
class Main_model():
    def __init__(self, n_vocab, dim_word, dim_ctx, dim):
        self.n_vocab = n_vocab
        self.dim_word = dim_word
        self.dim_ctx = dim_ctx
        self.dim = dim

        ### Word Embedding ###
        self.Wemb = initializations.uniform((n_vocab, self.dim_word))

        ### LSTM initialization NN ###
        self.Init_state_W = initializations.uniform((self.dim_ctx, self.dim))
        self.Init_state_b = shared_zeros((self.dim))

        self.Init_memory_W = initializations.uniform((self.dim_ctx, self.dim))
        self.Init_memory_b = shared_zeros((self.dim))


        ### Main LSTM ###
        self.lstm_W = initializations.uniform((self.dim_word, self.dim * 4))
        self.lstm_U = sharedX(np.concatenate([ortho_weight(dim),
                                      ortho_weight(dim),
                                      ortho_weight(dim),
                                      ortho_weight(dim)], axis=1))

        self.lstm_b = shared_zeros((self.dim*4))

        self.Wc = initializations.uniform((self.dim_ctx, self.dim*4)) # image -> LSTM hidden
        self.Wc_att = initializations.uniform((self.dim_ctx, self.dim_ctx)) # image -> 뉴럴넷 한번 돌린것
        self.Wd_att = initializations.uniform((self.dim, self.dim_ctx)) # LSTM hidden -> image에 영향
        self.b_att = shared_zeros((self.dim_ctx))

        self.U_att = initializations.uniform((self.dim_ctx, 1)) # image 512개 feature 1차원으로 줄임
        self.c_att = shared_zeros((1))

        ### Decoding NeuralNets ###
        self.decode_lstm_W = initializations.uniform((self.dim, self.dim_word))
        self.decode_lstm_b = shared_zeros((self.dim_word))

        self.decode_word_W = initializations.uniform((self.dim_word, n_vocab))
        self.decode_word_b = shared_zeros((n_vocab))

        self.params = [self.Wemb,
                       self.Init_state_W, self.Init_state_b,
                       self.Init_memory_W, self.Init_memory_b,
                       self.lstm_W, self.lstm_U, self.lstm_b,
                       self.Wc, self.Wc_att, self.Wd_att, self.b_att,
                       self.U_att, self.c_att,
                       self.decode_lstm_W, self.decode_lstm_b,
                       self.decode_word_W, self.decode_word_b]

        self.param_names = ['Wemb', 'Init_state_W', 'Init_state_b',
                            'Init_memory_W', 'Init_memory_b',
                            'lstm_W', 'lstm_U', 'lstm_b',
                            'Wc', 'Wc_att', 'Wd_att', 'b_att',
                            'U_att', 'c_att',
                            'decode_lstm_W', 'decode_lstm_b',
                            'decode_word_W', 'decode_word_b']


    def get_initial_lstm(self, ctx_mean):
        initial_state = T.dot(ctx_mean, self.Init_state_W) + self.Init_state_b # (n_samples, dim)
        initial_memory = T.dot(ctx_mean, self.Init_memory_W) + self.Init_memory_b # (n_samples, dim)

        return initial_state, initial_memory

    def forward_lstm(self, ctx, emb, mask, initial_state, initial_memory, one_step=False):

        if one_step:
            assert initial_state, "previous state must be provided"

        if emb.ndim == 3: #(n_timesteps, n_samples, dim)
            n_samples = emb.shape[1]
        else:
            n_samples = 1


        if mask is None:
            mask_shuffled = T.alloc(1., emb.shape[0], 1)

        else:
            mask_shuffled = mask.dimshuffle(1,0) # (n_samples, n_timesteps) => (n_timesteps, n_samples)


        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:,:,n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        def _step(m_tm_1, x_t, h_tm_1, c_tm_1, alpha_tm_1, alpha_sample_tm_1, pctx):

            # m_tm_1 : (n_samples, 1)
            # x_t : (n_samples, dim)
            # h_tm_1 : (n_samples, dim)
            # c_tm_1 : (n_samples, dim)
            # alpha_tm_1 : (n_samples, 196)
            # alpha_sample_tm_1 : (n_samples, 196)
            # att_ctx_tm_1 :  (n_samples, 512)
            # 근데 사실상 함수 내에서 쓰이는 변수는 m_tm_1, x_t, h_tm_1, c_tm_1 뿐임.
            # 나머지는 그냥 각 step마다 return만 됨. (outputs_info에 포함되어 있어서 강제로 input에 포함)

            projected_ctx = pctx + T.dot(h_tm_1, self.Wd_att)[:,None,:] + self.b_att # (n_samples, 196, 512)
            projected_ctx = T.tanh(projected_ctx)

            alpha = T.dot(projected_ctx, self.U_att) + self.c_att # (n_samples, 196, 1)
            alpha_shape = alpha.shape

            # 귀찮으니 일단 deterministic attention만 구현한다
            alpha = softmax(alpha.reshape([alpha_shape[0], alpha_shape[1]])) # 마지막 dimension 없앰
            weighted_ctx = (ctx * alpha[:,:,None]).sum(1) # (n_samples, 196, 512) * (n_samples, 196, 1)
            alpha_sample = alpha

            lstm_preact = T.dot(h_tm_1, self.lstm_U) + x_t + T.dot(weighted_ctx, self.Wc) # (n_samples, dim*4)
            i = T.nnet.sigmoid(_slice(lstm_preact, 0, self.dim)) # (n_samples, dim)
            f = T.nnet.sigmoid(_slice(lstm_preact, 1, self.dim)) # (n_samples, dim)
            o = T.nnet.sigmoid(_slice(lstm_preact, 2, self.dim)) # (n_samples, dim)
            c = T.tanh(_slice(lstm_preact, 3, self.dim)) # (n_samples, dim)

            c = f * c_tm_1 + i * c # (n_samples, dim)
            c = m_tm_1[:, None] * c + (1. - m_tm_1)[:,None] * c_tm_1 # (n_samples, dim)

            h = o * T.tanh(c) # (n_samples, dim)
            h = m_tm_1[:, None] * h + (1. - m_tm_1)[:,None] * h_tm_1 # (n_samples, dim)


            return [h, c, alpha, alpha_sample, weighted_ctx]

        projected_ctx = T.dot(ctx, self.Wc_att)
        X_t = T.dot(emb, self.lstm_W) + self.lstm_b # (n_timesteps, n_samples, dim*4)

        alpha_init = T.alloc(0., n_samples, ctx.shape[1]) # (n_samples, 196)
        alpha_sample_init = T.alloc(0., n_samples, ctx.shape[1]) # (n_samples, 196)

        sequences = [mask_shuffled, X_t]

        if one_step:
            rval = _step(mask_shuffled, X_t, initial_state, initial_memory, None, None, projected_ctx)

        else:

            outputs_info = [
                initial_state,
                initial_memory,
                alpha_init,
                alpha_sample_init,
                None
                ]

            rval, updates = theano.scan(_step,
                                        sequences=sequences,
                                        outputs_info=outputs_info,
                                        non_sequences=[projected_ctx])

        return rval

    def build_model(self):

        x = T.imatrix('x') # (n_samples, n_timesteps)
        mask = T.matrix('mask',dtype='float32') # (n_samples, n_timesteps)
        ctx = T.tensor3('ctx',dtype='float32') # (n_samples, 196, 512)

        initial_state, initial_memory = self.get_initial_lstm(ctx.mean(axis=1))

        emb = self.Wemb[x] # (n_samples, n_timesteps, dim_word)
        emb = emb.dimshuffle(1,0,2) # (n_timesteps, n_samples, dim_word)

        emb_shifted = T.zeros_like(emb)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1]) # 맨 앞에 0을 padding함. 예측해야되니까
        emb = emb_shifted

        rval = self.forward_lstm(ctx, emb, mask, initial_state, initial_memory)

        hiddens, cells, alphas, alpha_samples, weighted_ctxs = rval

        decoded_word_vec = T.dot(hiddens, self.decode_lstm_W) + self.decode_lstm_b
        decoded_word_vec = T.tanh(decoded_word_vec)
        decoded_word = T.dot(decoded_word_vec, self.decode_word_W) + self.decode_word_b
        decoded_word = decoded_word.dimshuffle(1,0,2)

        decoded_word_shape = decoded_word.shape
        probs = softmax(decoded_word.reshape([decoded_word_shape[0]*decoded_word_shape[1], decoded_word_shape[2]]))

        x_flat = x.flatten() # x_flat: [1   27  39  10  ...]
        p_flat = probs.flatten() # p_flat: [1 => 0100000..., 27 => 00000...1000..., 39 => 00000...00100...] 이런식
        cost = -T.log(p_flat[T.arange(x_flat.shape[0])*probs.shape[1] + x_flat] + 1e-8)
        # x_flat.shape[0] : n_samples * n_timesteps. arange()하니까 (0 ~ n_samples*n_timesteps - 1)
        # probs.shape[1] : n_vocab

        cost = cost.reshape([x.shape[0], x.shape[1]])
        masked_cost = cost * mask
        #cost = (masked_cost).sum(0)
        cost = (masked_cost).sum() / mask.sum()

        return x, mask, ctx, alphas, alpha_samples, cost, rval#, masked_cost

    def build_sampling_function(self):
        ctx = T.matrix('ctx')

        initial_state, initial_memory = self.get_initial_lstm(ctx.mean(axis=0))
        f_init = theano.function(inputs=[ctx],
                                 outputs=[ctx, initial_state, initial_memory])

        ctx = T.matrix('ctx')
        x = T.ivector('x')
        emb = T.switch(x[:,None] < 0, T.alloc(0., 1, self.Wemb.shape[1]), self.Wemb[x])

        initial_state = T.vector('initial_state')
        initial_memory = T.vector('initial_memory')

        [
            next_state,
            next_memory,
            alphas,
            alpha_samples,
            weighted_ctxs
        ] =  self.forward_lstm(ctx,
                               emb,
                               None,
                               initial_state[None, :],
                               initial_memory[None, :],
                               one_step=True)


        hiddens = next_state

        decoded_word_vec = T.dot(hiddens, self.decode_lstm_W) + self.decode_lstm_b
        decoded_word_vec = T.tanh(decoded_word_vec)

        decoded_word = T.dot(decoded_word_vec, self.decode_word_W) + self.decode_word_b

        next_probs = T.nnet.softmax(decoded_word)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        f_next = theano.function(inputs=[x, ctx, initial_state, initial_memory],
                                 outputs=[next_probs, next_sample, next_state, next_memory],
                                 allow_input_downcast=True)

        return f_init, f_next

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):

    grads = T.grad(cost=cost, wrt=params)
    updates = []

    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

    return updates

def sgd(cost, params, lr=0.001):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append([param, param-grad*lr])
    return updates

def train():

    data_path = '/home/taeksoo/Study/show_attend_and_tell/data/flickr30k'
    image_path = '/home/taeksoo/Study/show_attend_and_tell/images/'
    annotation_path = os.path.join(data_path, 'results_20130124.token')
    flickr_image_path = os.path.join(image_path, 'flickr30k-images')
    dictionary_path = os.path.join(data_path, 'dictionary.pkl')

    n_vocab = 10000
    dim_word = 512
    dim_ctx = 512
    dim = 512
    alpha_c = 0# 0.01 # alpha의 합이 1이 되도록 regularization
    decay_c = 0.0001# l2 regularization
    n_epochs = 10
    learning_rate = 0.001

    vgg_model = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    vgg_deploy = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

    cnn = CNN(
            deploy=vgg_deploy,
            model=vgg_model,
            batch_size=20,
            width=224,
            height=224)


    dictionary = pd.read_pickle(dictionary_path)

    #main_model = Main_model(n_vocab, dim_word, dim_ctx, dim)
    with open('./cv/iter_9.pickle') as f:
        main_model = cPickle.load(f)

    x, mask, ctx, alphas, alpha_samples, cost_theano, rval = main_model.build_model()
    f_init, f_next = main_model.build_sampling_function()

    if alpha_c > 0.:
        alpha_c = theano.shared(np.float32(alpha_c))
        alpha_reg = alpha_c * ((1. - alphas.sum(axis=0))**2).sum(axis=0).mean()
        cost_theano += alpha_reg

    if decay_c > 0.:
        decay_c = theano.shared(np.float32(decay_c))
        weight_decay = 0.
        for p in main_model.params:
            weight_decay += (p ** 2).sum()
        weight_decay *= decay_c
        cost_theano += weight_decay

    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))
    images = annotations['image'].values
    captions = annotations['caption'].values

    index = np.arange(len(images))
    np.random.shuffle(index)

    images = images[index]
    captions = captions[index]

    for epoch in range(n_epochs):
        updates = RMSprop(cost=cost_theano, params=main_model.params, lr=learning_rate)
        train_function = theano.function(inputs=[x,mask,ctx],
                                         outputs=cost_theano,
                                         updates=updates,
                                         allow_input_downcast=True,
                                         )

        for start, end in zip(range(0, len(images)+100, 100), range(100, len(images)+100, 100)):
            current_sents = captions[start:end]
            current_sent_ind = map(lambda sent: map(lambda word: dictionary[word] if word in dictionary else 1, sent.lower().split(' ')[:-1]), current_sents)

            current_imgs = images[start:end]
            current_feats = cnn.get_features(current_imgs, layers='conv5_3', layer_sizes=[512,14,14]).reshape(-1, 512, 196).swapaxes(1,2)

            maxlen = np.max(map(lambda x: len(x), current_sent_ind)) + 1

            X_train = sequence.pad_sequences(current_sent_ind, padding='post', maxlen=maxlen)
            mask_train = np.zeros_like(X_train)# * (1 - np.equal(X_train, 0))

            nonzeros = np.array(map(lambda x: (x != 0).sum(), X_train))

            for ind,row in enumerate(mask_train):
                row[:nonzeros[ind]+1] = 1

            cost = train_function(X_train, mask_train, current_feats)
            print start, ':', cost

        with open('./cv/iter_'+str(epoch+10)+'.pickle', 'w') as f:
            cPickle.dump(main_model, f)

        learning_rate *= 0.98

def gen_sample(model, ctx0):
    f_init, f_next = model.build_sampling_function()


