# -*- coding: utf-8 -*-
# 交易时间,交易金额,转账附言,渠道,发起方id,发起方年龄,发起方所处地区,接收方ID
import numpy as np
import time
import networkx as nx
import tensorflow as tf
from sklearn.ensemble import IsolationForest


def trans_time(dat, col=0):
    lis = map(lambda x: list(time.strptime(x, "%Y-%m-%d %H:%M:%S")), dat[:, col])
    arr_inner = np.array(lis)
    return arr_inner[:, :6]


def txt2id(dat, col=2, name=None):
    val = set(dat[:, col])
    dic = dict(zip(val, range(len(val))))
    arr = np.array([dic[item] for item in dat[:, col]])
    return arr, len(val)


def id2node(dat, cols=[4, 7]):
    lis = [dat[i, cols[0]]+"_"+dat[i, cols[1]] for i in range(dat.shape[0])]
    return np.array(lis)


def create_g(dat, cols=[4, 7]):
    g = nx.Graph()
    edges = np.unique(dat[:, cols])
    g.add_edges_from(edges)
    return g


def compute_support(dat1):
    edge_graph = nx.Graph()
    [edge_graph.add_node(node) for node in range(dat1.shape[0])]
    for i in range(dat1.shape[0]):
        set1 = set(dat1[i, [4, 7]])
        for j in range(i+1, dat1.shape[0]):
            if (dat1[j, 4] in set1) or (dat1[j, 7] in set1):
                edge_graph.add_edge(i, j)
    adj_mx_ = nx.adjacency_matrix(edge_graph).toarray()
    edges_ = np.array(nx.edges(edge_graph))
    return adj_mx_, edges_


def negative_sampling_edges(_edges, num_node):
    sample_edges = []
    for _i in range(1):
        for row in range(num_node):
            node = _edges[row, 0]
            pos_node = _edges[row, 1]
            for _ in range(num_node):
                neg_node = np.random.randint(0, num_node)
                if (neg_node != node) and (neg_node != pos_node):
                    sample_edges.append([node, pos_node, neg_node])
                    sample_edges.append([pos_node, node, neg_node])
                    break
    tuple_edges_ = np.array(sample_edges)
    return tuple_edges_


def negative_sampling(support_):
    sample_edges = []
    num_node = support_.shape[0]
    for node in range(num_node):
        for k in range(num_node):
            if (k != node) and (support_[node, k] != 0.):
                pos_node = k
                for _ in range(num_node*2):
                    neg_node = np.random.randint(0, num_node)
                    if (neg_node != node) and (neg_node != pos_node):
                        sample_edges.append([node, pos_node, neg_node])
                        break
    tuple_edges_ = np.array(sample_edges)
    return tuple_edges_


class GraphModel(object):
    def __init__(self, tuple_edges, msg_size, way_size, address_size, batch_size=64, dense_size=8):
        with tf.variable_scope("input_layer"):
            self.dense = tf.placeholder(dtype=tf.float32, shape=[batch_size, dense_size])
            self.msg = tf.placeholder(dtype=tf.int32, shape=[batch_size])
            self.way = tf.placeholder(dtype=tf.int32, shape=[batch_size])
            self.address = tf.placeholder(dtype=tf.int32, shape=[batch_size])
            self.support = tf.placeholder(dtype=tf.float32, shape=[batch_size, batch_size])

        with tf.variable_scope("initialize_LC_embedding"):
            self.msg_LC = tf.get_variable("msg", [msg_size, int(np.sqrt(msg_size))], dtype=tf.float32,
                                          initializer=tf.random_normal_initializer)
            self.way_LC = tf.get_variable("way", [way_size, int(np.sqrt(way_size))], dtype=tf.float32,
                                          initializer=tf.random_normal_initializer)
            self.address_LC = tf.get_variable("address", [address_size, int(np.sqrt(address_size))], dtype=tf.float32,
                                              initializer=tf.random_normal_initializer)

        with tf.variable_scope("LC_embedding"):
            self.msg_embed = tf.nn.embedding_lookup(self.msg_LC, self.msg)
            self.way_embed = tf.nn.embedding_lookup(self.way_LC, self.way)
            self.address_embed = tf.nn.embedding_lookup(self.address_LC, self.address)

        with tf.variable_scope("concat_layer"):
            self.input_concat = tf.concat([self.dense, self.msg_embed, self.way_embed, self.address_embed], axis=1)

        with tf.variable_scope("hidden_layer_1st"):
            self.h0 = tf.matmul(self.support, self.input_concat)
            d1 = self.input_concat.shape[1].value
            d1_ = max(10, int(np.sqrt(d1)))
            w1 = tf.get_variable("hidden1", [d1, d1_], initializer=tf.random_normal_initializer)
            self.h0_ = tf.nn.elu(tf.matmul(self.h0, w1))

        with tf.variable_scope("hidden_layer_2nd"):
            self.h1 = tf.matmul(self.support, self.h0_)
            d2 = self.h1.shape[1].value
            d2_ = max(10, int(np.sqrt(d2)))
            w2 = tf.get_variable("hidden2", [d2, d2_], initializer=tf.random_normal_initializer)
            self.h1_ = tf.nn.elu(tf.matmul(self.h1, w2))

        with tf.variable_scope("hidden_layer_3rd"):
            self.h2 = tf.matmul(self.support, self.h1_)
            d3 = self.h2.shape[1].value
            d3_ = max(10, int(np.sqrt(d3)))
            w3 = tf.get_variable("hidden3", [d3, d3_], initializer=tf.random_normal_initializer)
            self.h3_ = tf.nn.elu(tf.matmul(self.h2, w3))

        with tf.variable_scope("output_layer"):
            self.output = self.h3_

        with tf.variable_scope("loss_layer"):
            # self.loss = tf.nn.l2_loss(self.output)
            self.loss = self.compute_loss(tuple_edges)

    def compute_loss(self, tuple_edges_):
        sample_edges = np.int64(tuple_edges_)
        output = self.output

        def _loss_fcn(output, sample_edges):
            loss_ = 0
            for n_id in range(sample_edges.shape[0]):
                nodes_ = sample_edges[n_id, 0]
                output_nodes = output[nodes_]
                pos_nodes = sample_edges[n_id, 1]
                neg_nodes = sample_edges[n_id, 2]
                print(nodes_, pos_nodes, neg_nodes)
                p1 = tf.reduce_sum(tf.multiply(output_nodes, output[pos_nodes]))
                p1 = tf.log(tf.sigmoid(p1) + 0.001)

                p2 = tf.reduce_sum(tf.multiply(output_nodes, output[neg_nodes]))
                p2 = tf.log(tf.sigmoid(-p2) + 0.001)

                loss_ += (p1 + 0.1*p2)
                return loss_

        loss_ = _loss_fcn(output, sample_edges)
        return loss_


def load_graph_data(dat1):
    st = time.time()

    # support
    support_, edges_ = compute_support(dat1)
    support_ = 10*np.eye(support_.shape[0]) + support_
    print("support", support_.shape)
    # sample_edges
    # tuple_edges_ = negative_sampling(support_)
    tuple_edges_ = negative_sampling_edges(edges_, support_.shape[0])
    print("tuple_edges", tuple_edges_.shape)
    # id_id
    # ind = id2node(dat1, cols=[4, 7])
    # 交易时间 len=6
    arr0 = trans_time(dat1, 0)
    print("time", arr0.shape)
    # 交易金额 len=1
    arr1 = np.expand_dims(np.int32(dat1[:, 1]), axis=-1)
    print("count", arr1.shape)
    # 转账附言
    arr2, msg_size = txt2id(dat1, 2)
    print("msg", arr2.shape)
    # 渠道
    arr3, way_size = txt2id(dat1, 3)
    print("way", arr3.shape)
    # 发起方年龄 len=1
    arr5_ = np.int32(dat1[:, 5])
    arr5 = np.expand_dims(arr5_, axis=-1)
    print("age", arr5.shape)
    # 发起方所处地区
    arr6, address_size = txt2id(dat1, 6)
    print("address", arr6.shape)

    dense_ = np.concatenate((arr0, arr1, arr5), axis=1)

    msg_ = arr2
    way_ = arr3
    address_ = arr6
    print("load graph data runtime %.5f" % (time.time()-st))
    return dense_, msg_, way_, address_, support_, msg_size, way_size, address_size, tuple_edges_


if __name__ == "__main__":
    file = "sample.txt"
    dat2 = np.loadtxt(file, skiprows=1, delimiter=",", dtype=str)
    dense, msg, way, address, support, msg_size, way_size, address_size, tuple_edges = load_graph_data(dat2)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            model = GraphModel(tuple_edges, msg_size, way_size, address_size, dense.shape[0], dense.shape[1])
            train_op = tf.train.AdamOptimizer(0.001).minimize(model.loss)
            sess.run(tf.global_variables_initializer())

            feed_dict = {
                model.dense: dense,
                model.way: way,
                model.msg: msg,
                model.address: address,
                model.support: support,
            }

            loss, embed = sess.run([model.loss, model.output], feed_dict=feed_dict)
            print("loss:", loss)
            np.savetxt("embed.txt", embed)
    rng = np.random.RandomState(0)
    clf = IsolationForest(random_state=rng, contamination=.0001)
    clf.fit(embed)
    result = clf.predict(embed)
    abnormal = dat2[result == -1]
    print("num of abnormal: %d", abnormal.shape[0])
    np.savetxt("abnormal.txt", abnormal, delimiter=",", fmt="%s")

