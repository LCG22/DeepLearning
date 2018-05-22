#! /usr/bin/env python
# coding: utf-8


"""
author: lcj22
date: 2018-05-15

"""
import numpy as np
import collections
import random
import tensorflow as tf
import math
import pandas as pd
import time


start_time = time.time()


def get_datas():

    datas_df = pd.read_csv("output_datas/datas_df.csv", encoding="utf-8")
    count_df = pd.read_csv("output_datas/count_df.csv", encoding="utf-8")
    dictionary_df = pd.read_csv("output_datas/dictionary_df.csv", encoding="utf-8")

    count_df["cnt"] = count_df["cnt"].astype(np.int32)
    dictionary_df["number"] = dictionary_df["number"].astype(np.int32)

    datas_list = datas_df["number"].tolist()
    count_dict = {count_df["word"][i_int]: count_df["cnt"][i_int] for i_int in count_df.index}
    dictionary_dict = {dictionary_df["word"][i_int]: dictionary_df["number"][i_int] for i_int in dictionary_df.index}
    reverse_dictionary_dict = {dictionary_df["number"][i_int]: dictionary_df["word"][i_int] for i_int in
                               dictionary_df.index}

    return datas_list, count_dict, dictionary_dict, reverse_dictionary_dict


data, count, dictionary_dict, reverse_dictionary = get_datas()

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):

    """

    :param batch_size: 每批生成多少个样本
    :param num_skips: 每个单词必须生成多少个样本，且 batch_size 必须是 num_skips 的整数倍
    :param skip_window: 单词最远可以联系的距离
    :return:
    """

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span -1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print batch[i], reverse_dictionary[batch[i]], "->", labels[i, 0], reverse_dictionary[labels[i, 0]]


valid_word = [u"买", u"楼", u"博士", u"儿子", u"睇房", u"中意", u"仲意", u"钟意", u"睇", u"得闲", u"倾成", u"几好",
              u"价钱", u"唔系"]
valid_word = [word for word in valid_word if word in dictionary_dict.keys()]
valid_examples = [dictionary_dict[li] for li in valid_word]
vocabulary_size = 50000
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2
valid_size = len(valid_word)
valid_window = 100
num_sampled = 64


# 用于记录参数
args_dict = {
    "vocabulary_size": vocabulary_size,
    "batch_size": batch_size,
    "embedding_size": embedding_size,
    "skip_window": skip_window,
    "num_skips": num_skips,
    "valid_size": valid_size,
    "valid_window": valid_window,
    "num_sampled": num_sampled
}

with open("output_datas/train_log.txt", "a") as f:
    f.write("训练参数如下：\n")
    f.write("{}".format(args_dict))


graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device("/cpu:0"):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        saver = tf.train.Saver()

    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(
        embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         inputs=embed,
                                         labels=train_labels,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size
                                         )
                          )

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 2000000
with tf.Session(graph=graph) as session:
    init.run()
    print "Initialized"

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print "Average loss at step", step, ": ", average_loss
            with open("output_datas/train_log.txt", "a") as f:
                f.write("信息结果相关信息如下：\n")
                f.write("Average loss at step %s：%s \n" % (step, average_loss))
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = "Nearest to %s: " % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s, " % (log_str, close_word)
                print log_str

                # 记录训练结果
                with open("output_datas/train_log.txt", "a") as f:
                    f.write("信息结果相关信息如下：\n")
                    f.write("词嵌入：\n")
                    f.write(repr(log_str))
            # 保存模型
            saver.save(session, "output_datas/model/model.ckpt")
            print "模型保存成功！"
            with open("output_datas/train_log.txt", "a") as f:
                f.write("模型保存成功！")
    # 保存模型
    saver.save(session, "output_datas/model/model.ckpt")
    print "模型保存成功！"
    with open("output_datas/train_log.txt", "a") as f:
        f.write("模型保存成功！")

    final_embeddings = normalized_embeddings.eval()


end_time = time.time()
cost_time = end_time - start_time
cost_time_str = "共花费了 {} 秒".format(cost_time)
print cost_time_str
with open("output_datas/train_log.txt", "a") as f:
    f.write(cost_time_str)





