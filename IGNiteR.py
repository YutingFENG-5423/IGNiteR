#-*-coding:utf-8-*-
import numpy as np
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten, Activation, Layer
from keras import backend as K
import keras
import tensorflow as tf
import pickle
from utils import dcg_score,ndcg_score,mrr_score, load_pkl
from sklearn.metrics import roc_auc_score,f1_score


max_words_length = 10
max_nodes_length = 30
max_adopt_length = 120
dict_length = 7275
node_embedding_dimension = 50
word_embedding_dimension = 300
click_history = 20
n_candidates = 5




news_words,embedding_mat,node_embedding_mat,\
all_user_pos,all_user_pos_nodes,all_user_pos_adopt,all_train_pn,all_train_nodes,all_train_adopt,all_label,all_train_id,\
all_test_user_pos,all_test_user_pos_nodes,all_test_user_pos_adopt,all_test_pn,all_test_nodes,all_test_adopt,all_test_label,all_test_id, all_test_index = load_pkl('/home/yuting/PycharmProjects/data_preprocessing/data_stat/weighted graph/data_T_3m_nb10_120.pkl')


title_input = keras.Input(shape=(max_words_length,), dtype='int32')
embedded_title = keras.layers.Embedding(len(embedding_mat),word_embedding_dimension,weights=[embedding_mat],trainable=True)(title_input)
title_cnn = keras.layers.Convolution1D(nb_filter=128, filter_length=3, padding='same', activation='relu', strides=1)(embedded_title)
title_cnn = keras.layers.Dropout(0.2)(title_cnn)

attention_title = keras.layers.Dense(200, activation='tanh')(title_cnn)
attention_title = Flatten()(Dense(1)(attention_title))
attention_weight_title = Activation('softmax',name='attention_title')(attention_title)
title_rep = keras.layers.Dot((1, 1))([title_cnn, attention_weight_title])


node_input = keras.Input(shape=(max_nodes_length,), dtype='int32')
embedded_node = keras.layers.Embedding(len(node_embedding_mat),node_embedding_dimension,weights=[node_embedding_mat],trainable=True)(node_input)
node_lstm = keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True))(embedded_node)
node_lstm = keras.layers.Dropout(0.2)(node_lstm)

attention_node = keras.layers.Dense(200, activation='tanh')(node_lstm)
attention_node = Flatten()(Dense(1)(attention_node))
attention_weight_node= Activation('softmax',name='attention_node')(attention_node)
node_rep = keras.layers.Dot((1, 1))([node_lstm, attention_weight_node])


# output batch_size * 128
adopt_input = keras.Input(shape=(max_adopt_length,), dtype='float32')
adopt = keras.layers.Dense(128, activation='tanh')(adopt_input)




channels = [title_rep,node_rep, adopt]

views = keras.layers.concatenate([keras.layers.Lambda(lambda x: K.expand_dims(x,axis=1))(channel) for channel in channels], axis=1)

attention_news = keras.layers.Dense(200, activation='tanh')(views)
attention_weight_news = keras.layers.Lambda(lambda x: K.squeeze(x, axis=-1))(Dense(1)(attention_news))
attention_weight_news = Activation('softmax',name="attention_view")(attention_weight_news)
news_rep = keras.layers.Dot((1, 1))([views, attention_weight_news])

newsEncoder = keras.Model([title_input,node_input,adopt_input],news_rep)
# newsEncoder.summary()


browsed_words_input = [keras.Input((max_words_length,), dtype='int32') for _ in range(click_history)]
browsed_nodes_input = [keras.Input((max_nodes_length,), dtype='int32') for _ in range(click_history)]
browsed_adopt_input = [keras.Input((max_adopt_length,), dtype='float32') for _ in range(click_history)]
browsednews = [newsEncoder([browsed_words_input[_],browsed_nodes_input[_],browsed_adopt_input[_]]) for _ in range(click_history)]
browsednewsrep = keras.layers.concatenate([keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in browsednews], axis=1)

browsednewsrep_lstm = keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True))(browsednewsrep)
browsednewsrep_lstm = keras.layers.Dropout(0.2)(browsednewsrep_lstm)

attention_history = keras.layers.Dense(200, activation='tanh')(browsednewsrep_lstm)
attention_history = Flatten()(Dense(1)(attention_history))
attention_weight_history= Activation('softmax',name='attention_history')(attention_history)
user_rep = keras.layers.Dot((1, 1),name="dot1")([browsednewsrep_lstm, attention_weight_history])


candidates_words = [keras.Input((max_words_length,), dtype='int32') for _ in range(n_candidates)]
candidates_nodes = [keras.Input((max_nodes_length,), dtype='int32') for _ in range(n_candidates)]
candidates_adopt = [keras.Input((max_adopt_length,), dtype='float32') for _ in range(n_candidates)]
candidate_vecs = [newsEncoder([candidates_words[_], candidates_nodes[_], candidates_adopt[_]]) for _ in range(n_candidates)]

logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))

model = keras.Model(candidates_words + browsed_words_input + candidates_nodes + browsed_nodes_input + candidates_adopt + browsed_adopt_input, logits)
# model.summary()

candidate_one_words = keras.Input((max_words_length,))
candidate_one_nodes = keras.Input((max_nodes_length,))
candidate_one_adopt = keras.Input((max_adopt_length,))

candidate_one_vec = newsEncoder([candidate_one_words, candidate_one_nodes, candidate_one_adopt])

score = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.dot([user_rep, candidate_one_vec], axes=-1))
model_test = keras.Model([candidate_one_words] + browsed_words_input + [candidate_one_nodes] + browsed_nodes_input + [candidate_one_adopt] + browsed_adopt_input, score)
# model_test.summary()
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['acc'])


def generate_batch_data_train(all_train_pn, all_train_nodes, all_train_adopt, all_label, batch_size):
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate_words = news_words[all_train_pn[i]]
            candidate_words_split = [candidate_words[:, k, :] for k in range(candidate_words.shape[1])]

            candidate_nodes = all_train_nodes[i]
            candidate_nodes_split = [candidate_nodes[:, k, :] for k in range(candidate_nodes.shape[1])]

            candidate_adopt = all_train_adopt[i]
            candidate_adopt_split = [candidate_adopt[:, k, :] for k in range(candidate_adopt.shape[1])]

            browsed_news_words = news_words[all_user_pos[i]]
            browsed_news_words_split = [browsed_news_words[:, k, :] for k in range(browsed_news_words.shape[1])]

            browsed_news_nodes = all_user_pos_nodes[i]
            browsed_news_nodes_split = [browsed_news_nodes[:, k, :] for k in range(browsed_news_nodes.shape[1])]

            browsed_news_adopt = all_user_pos_adopt[i]
            browsed_news_adopt_split = [browsed_news_adopt[:, k, :] for k in range(browsed_news_adopt.shape[1])]

            label = all_label[i]

            yield (candidate_words_split + browsed_news_words_split + candidate_nodes_split + browsed_news_nodes_split + candidate_adopt_split + browsed_news_adopt_split, [label])


def generate_batch_data_test(all_test_pn, all_test_nodes, all_test_adopt, all_label, batch_size):
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate_words = news_words[all_test_pn[i]]
            candidate_nodes = all_test_nodes[i]
            candidate_adopt = all_test_adopt[i]

            browsed_news_words = news_words[all_test_user_pos[i]]
            browsed_news_words_split = [browsed_news_words[:, k, :] for k in range(browsed_news_words.shape[1])]
            browsed_news_nodes = all_test_user_pos_nodes[i]
            browsed_news_nodes_split = [browsed_news_nodes[:, k, :] for k in range(browsed_news_nodes.shape[1])]
            browsed_news_adopt = all_test_user_pos_adopt[i]
            browsed_news_adopt_split = [browsed_news_adopt[:, k, :] for k in range(browsed_news_adopt.shape[1])]

            label = all_label[i]
            yield ([candidate_words] + browsed_news_words_split + [candidate_nodes] + browsed_news_nodes_split + [candidate_adopt] + browsed_news_adopt_split, [label])


results = []
batch_size = 100

for ep in range(3):
    traingen = generate_batch_data_train(all_train_pn, all_train_nodes, all_train_adopt, all_label, all_train_id, batch_size)
    model.fit_generator(traingen, epochs=1, steps_per_epoch=len(all_train_id) // batch_size)
    testgen = generate_batch_data_test(all_test_pn, all_test_nodes, all_test_adopt, all_test_label, all_test_id, batch_size)
    click_score = model_test.predict_generator(testgen, steps=len(all_test_id) // batch_size, verbose=1)


    all_auc = []
    all_mrr = []
    all_ndcg = []
    all_ndcg2 = []
    all_f1 = []
    for m in all_test_index:
        if np.sum(all_test_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
            all_auc.append(roc_auc_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
            all_f1.append(f1_score(all_test_label[m[0]:m[1]], np.round(abs(click_score[m[0]:m[1], 0]))))
            all_mrr.append(mrr_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
            all_ndcg.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=5))
            all_ndcg2.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=10))
    results.append([np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg), np.mean(all_ndcg2)])
    print(np.mean(all_auc), np.mean(all_f1), np.mean(all_mrr), np.mean(all_ndcg), np.mean(all_ndcg2))





