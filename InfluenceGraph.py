#-*-coding:utf-8-*-

import numpy as np
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten, Activation, Layer
import keras
import tensorflow as tf
import pandas as pd



def get_nodes_dict(filname):
    f = open(filname + "/train_set.txt", "r")
    initiators = []
    for l in f:
        parts = l.split(",")
        initiators.append(parts[0])
    # ----------------- Source node dictionary
    initiators = np.unique((initiators))
    dict_in = {initiators[i]: i for i in range(0, len(initiators))}
    f.close()

    vocabulary_size = len(dict_in)
    print(vocabulary_size)
    # ----------------- Target node dictionary
    graph = pd.read_csv(filname + "_network.txt", sep=" ")
    graph.columns = ["node1", "node2", "weight"]
    all = list(set(graph["node1"].unique()).union(set(graph["node2"].unique())))
    dict_out = {int(all[i]): i for i in range(0, len(all))}
    target_size = len(dict_out)
    print(target_size)

    return dict_in,dict_out


filename = 'home/InflenceGraph/data/weibo'
dict_in, dict_out = get_nodes_dict(filename)

vocabulary_size = len(dict_in)
embedding_size = 50
target_size = len(dict_out)
num_samples = 10
init_weights = np.random.normal(loc=0,scale=1e-2,size=(target_size,embedding_size))


u_input = keras.Input(shape=(1,), dtype='int32')
v_input = keras.Input(shape=(1,), dtype='int32')
t_input = keras.Input(shape=(target_size,), dtype='int32')

S = keras.layers.Embedding(vocabulary_size,embedding_size,embeddings_initializer="uniform",trainable=True,name='S')
u_embedding = S(u_input)

initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
T = keras.layers.Embedding(target_size,embedding_size,embeddings_initializer=initializer,trainable=True, name='T')

embeddings = T(t_input)

biases = Embedding(target_size, 1)(t_input)
biases = Flatten()(biases)

logits = keras.layers.Dot(axes=(2))([u_embedding,embeddings])
logits = Flatten()(logits)
logits = keras.layers.Add()([logits,biases])

output = keras.layers.Activation(keras.activations.softmax)(logits)

model = keras.Model([u_input,v_input,t_input],output)

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['acc'])
# model_embedding = keras.Model(inputs=model.input, outputs=model.get_)

model.summary()



for epoch in range(3):
    f = open(filename+ "/train_set.txt", "r")
    idx = 0
    init = -1
    inputs = []
    labels = []
    # ---- Build the input batch
    for line in f:
        # ---- input node, output node, copying_time, cascade_length, 10 negative samples
        sample = line.replace("\r", "").replace("\n", "").split(",")
        try:
            original = dict_in[sample[0]]
            label = dict_out[sample[1]]
        except:
            continue
        # ---- check if we are at the same cascade
        if (init == original or init < 0):
            init = original
            inputs.append(original)
            labels.append(label)
        # ---- New cascade, train on the previous one
        else:
            # ---------- Run one training batch
            # --- Train for target nodes
            if len(inputs) < 2:
                inputs.append(inputs[0])
                labels.append(labels[0])
            inputs = np.asarray(inputs).reshape((len(inputs), 1))
            labels = np.asarray(labels).reshape((len(labels), 1))

            model.fit_generator(inputs,labels)

            idx += 1

            # ---- Arrange for the next batch
            inputs = []
            labels = []
            inputs.append(original)
            labels.append(label)
            init = original

file_s = filename + "/embeddings_initiators.txt"
file_t = filename + "/embeddings_targets.txt"
fsn = open(file_s, "w")
ftn = open(file_t, "w")

# ---------- Store the source embedding of each node
for node in dict_in.keys():
    emb_Sn = model.get_layer('S').get_weights()[0]
    fsn.write(node + ":" + ",".join([str(s) for s in list(emb_Sn)]) + "\n")
fsn.close()
# ---------- Store the target embedding of each node
for node in dict_out.keys():
    emb_Tn = model.get_layer('T').get_weights()[0]
    ftn.write(node + ":" + ",".join([str(s) for s in list(emb_Tn)]) + "\n")
ftn.close()