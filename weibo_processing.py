# -*- coding: utf-8 -*-
"""
Data from https://aminer.org/influencelocality
Extract network and diffusion cascades from Weibo
"""

import os
import time
import tarfile
import igraph as ig
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json


def split_train_and_test(cascades_file):
    """
    # Keeps the ids of the users that are actively retweeting
    # Train time:(2011.10.29 -2012.9.28) and test time (2012.9.28 -2012.10.29)
    """
    f = open(cascades_file)
    ids = set()
    train_cascades = []
    test_cascades = []
    counter = 0

    for line in f:

        date = line.split(" ")[1].split("-")
        original_user_id = line.split(" ")[2]

        retweets = f.__next__().replace(" \n", "").split(" ")
        # ----- keep only the cascades and the nodes that are active in train (2011.10.29 -2012.9.28) and test (2012.9.28 -2012.10.29)

        retweet_ids = ""

        # ------- last month at test       choose in 2012 after 09.28 or in 10, before 10.29 !!! for test
        if int(date[0]) == 2012 and (
                (int(date[1]) >= 9 and int(date[2]) >= 28) or (int(date[1]) == 10 and int(date[2]) <= 29)):
            ids.add(original_user_id)

            cascade = ""
            for i in range(0, len(retweets) - 1, 2):
                ids.add(retweets[i])
                retweet_ids = retweet_ids + " " + retweets[i]
                cascade = cascade + ";" + retweets[i] + " " + retweets[i + 1]

            # ------- For each cascade keep also the original user and the relative day of recording (1-32)
            date = str(int(date[2]) + 3)
            op = line.split(" ")
            op = op[2] + " " + op[1]
            test_cascades.append(date + ";" + op + cascade)

        # ------ The rest are used for training     choose in 2012 before 09.28 or 2011 after 10.29 for training
        elif (int(date[0]) == 2012 and int(date[1]) < 9 and int(date[2]) < 28) or (
                int(date[0]) == 2011 and int(date[1]) >= 10 and int(date[2]) >= 29):

            ids.add(original_user_id)
            cascade = ""
            for i in range(0, len(retweets) - 1, 2):
                ids.add(retweets[i])
                retweet_ids = retweet_ids + " " + retweets[i]
                cascade = cascade + ";" + retweets[i] + " " + retweets[i + 1]
            if (int(date[1]) == 9):
                date = str(int(date[2]) - 27)
            else:
                date = str(int(date[2]) + 3)
            op = line.split(" ")
            op = op[2] + " " + op[1]
            train_cascades.append(date + ";" + op + cascade)

        counter += 1
        if (counter % 10000 == 0):
            print("------------" + str(counter))
    f.close()

    return train_cascades, test_cascades, ids


def weibo_preprocessing(path):
    os.chdir(path)
    # download()
    filepath = '/data/'
    # ------ Split the original retweet cascades
    train_cascades, test_cascades, ids = split_train_and_test(filepath + 'total.txt')

    # ------ Store the cascades
    print("Size of train:", len(train_cascades))
    print("Size of test:", len(test_cascades))

    with open("train_cascades.txt", "w") as f:
        for cascade in train_cascades:
            f.write(cascade + "\n")

    with open("test_cascades.txt", "w") as f:
        for cascade in test_cascades:
            f.write(cascade + "\n")

    # ------ Store the active ids
    f = open("active_users.txt", "w")
    for uid in ids:
        f.write(uid + "\n")
    f.close()

    # ------ Keep the subnetwork of the active users
    g = open("weibo_network.txt", "w")

    f = open(filepath + "graph_170w_1month.txt", "r")

    found = 0
    idx = 0
    for line in f:
        edge = line.replace("\n", "").split(" ")

        if edge[0] in ids and edge[1] in ids and edge[2] == '1':
            found += 1
            g.write(line)
        idx += 1
        if (idx % 2000000 == 0):
            print(idx)
            print(found)
            print("---------")

    g.close()

    f.close()



"""
Compute kcore and avg cascade length
Extract the train set for INFECTOR
"""


def remove_duplicates(cascade_nodes, cascade_times):
    """
    # Some tweets have more then one retweets from the same person
    # Keep only the first retweet of that person
    """
    duplicates = set([x for x in cascade_nodes if cascade_nodes.count(x) > 1])
    for d in duplicates:
        to_remove = [v for v, b in enumerate(cascade_nodes) if b == d][1:]
        cascade_nodes = [b for v, b in enumerate(cascade_nodes) if v not in to_remove]
        cascade_times = [b for v, b in enumerate(cascade_times) if v not in to_remove]

    return cascade_nodes, cascade_times


def store_samples(fn, cascade_nodes, cascade_times, initiators, train_set, op_time, sampling_perc=120):
    """
    # Store the samples  for the train set as described in the node-context pair creation process for INFECTOR
    """
    # ---- Inverse sampling based on copying time
    # op_id = cascade_nodes[0]
    no_samples = round(len(cascade_nodes) * sampling_perc / 100)
    casc_len = len(cascade_nodes)
    # times = [op_time/(abs((cascade_times[i]-op_time))+1) for i in range(0,len(cascade_nodes))]
    times = [1.0 / (abs((cascade_times[i] - op_time)) + 1) for i in range(0, casc_len)]
    s_times = sum(times)

    if s_times == 0:
        samples = []
    else:
        print("out")
        probs = [float(i) / s_times for i in times]
        samples = np.random.choice(a=cascade_nodes, size=int(no_samples), p=probs)  # closer time, higher probability

    # ----- Store train set
    op_id = initiators[0]
    for i in samples:
        # if(op_id!=i):
        # ---- Write initial node, copying node, copying time, length of cascade
        train_set.write(str(op_id) + "," + i + "," + str(casc_len) + "\n")


def run(fn, sampling_perc, log=None):
    print("Reading the network")
    g = ig.Graph.Read_Ncol(fn + "_network.txt")


    f = open("train_cascades.txt", "r")
    train_set = open("train_set.txt", "w")

    # ----- Initialize features
    idx = 0
    deleted_nodes = []
    g.vs["Cascades_started"] = 0
    g.vs["Cumsize_cascades_started"] = 0
    g.vs["Cascades_participated"] = 0
    log.write(" net:" + fn + "\n")
    start_t = 0  # int(next(f))
    idx = 0

    start = time.time()
    # ---------------------- Iterate through cascades to create the train set
    for line in f:
        initiators = []
        cascade = line.replace("\n", "").split(";")
        if (fn == "weibo"):
            cascade_nodes = list(map(lambda x: x.split(" ")[0], cascade[1:]))
            # cascade_times = list(map(lambda x:  datetime.strptime(x.replace("\r","").split(" ")[1], '%Y-%m-%d-%H:%M:%S'),cascade[1:]))
            cascade_times = list(map(lambda x: int(((datetime.strptime(x.replace("\r", "").split(" ")[1],
                                                                       '%Y-%m-%d-%H:%M:%S') - datetime.strptime(
                "2011-10-28", "%Y-%m-%d")).total_seconds())), cascade[1:]))

        # ---- Remove retweets by the same person in one cascade
        cascade_nodes, cascade_times = remove_duplicates(cascade_nodes, cascade_times)

        # ---------- Dictionary nodes -> cascades
        op_id = cascade_nodes[0]  # first node original node
        op_time = cascade_times[0]

        # ---------- Update metrics
        try:
            g.vs.find(name=op_id)["Cascades_started"] += 1
            g.vs.find(op_id)["Cumsize_cascades_started"] += len(cascade_nodes)
        except:
            deleted_nodes.append(op_id)
            continue

        if (len(cascade_nodes) < 2):
            continue
        initiators = [op_id]
    store_samples(fn, cascade_nodes[1:], cascade_times[1:], initiators, train_set, op_time, sampling_perc)

    idx += 1
    if (idx % 1000 == 0):
        print("-------------------", idx)

    print("Number of nodes not found in the graph: ", len(deleted_nodes))
    f.close()
    train_set.close()
    log.write("Feature extraction time:" + str(time.time() - start) + "\n")

    kcores = g.shell_index()  # all the nodes
    log.write("K-core time:" + str(time.time() - start) + "\n")
    a = np.array(g.vs["Cumsize_cascades_started"], dtype=np.float)
    b = np.array(g.vs["Cascades_started"], dtype=np.float)

    # ------ Store node charateristics
    pd.DataFrame({"Node": g.vs["name"],
                  "Kcores": kcores,
                  "Participated": g.vs["Cascades_participated"],
                  "Avg_Cascade_Size": a / b}).to_csv(fn + "_node_features.csv", index=False)
    # "Avg_Cascade_Size": a/b}).to_csv(fn+"/node_features.csv",index=False)

    # ------ Derive incremental node dictionary
    graph = pd.read_csv(fn + "_network.txt", sep=" ")
    graph.columns = ["node1", "node2", "weight"]
    all = list(set(graph["node1"].unique()).union(set(graph["node2"].unique())))
    dic = {int(all[i]): i for i in range(0, len(all))}
    f = open(fn + "_incr_dic.json", "w")
    json.dump(dic, f)
    f.close()


# ==========================================================
if __name__ == '__main__':
    ## Create folder structure
    path = os.getcwd()
    ans = weibo_preprocessing(path)

    print(ans)

    fn = 'weibo'
    sampling_perc = 120

    run(fn, sampling_perc)





