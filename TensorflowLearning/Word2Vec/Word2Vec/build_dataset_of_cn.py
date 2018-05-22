#! /usr/bin/env python
# coding: utf-8


"""
将中文按照频数进行映射为数字

author: lcj22
date: 2018-05-15

"""


import collections
import pandas as pd


def build_dataset(words, size=5000):

    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(size - 1))

    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    words_df = pd.DataFrame()
    words_df["number"] = data
    words_df.to_csv("output_datas/datas_df.csv", index=False)

    count_df = pd.DataFrame()
    count_df["word"] = [val_tuple[0] for val_tuple in count]
    count_df["cnt"] = [val_tuple[1] for val_tuple in count]
    count_df.to_csv("output_datas/count_df.csv", index=False)

    dictionary_df = pd.DataFrame()
    dictionary_df["word"] = [key_str for key_str in dictionary.keys()]
    dictionary_df["number"] = [val_str for val_str in dictionary.values()]
    dictionary_df.to_csv("output_datas/dictionary_df.csv", index=False)

    print "数据保存成功！"

    return data, count, dictionary, reverse_dictionary


def main():

    read_filename_str = "output_datas/processed_datas_cn.csv"
    df = pd.read_csv(read_filename_str)
    words_list = df["word"].tolist()
    filename_str = "output_datas/datasset.py"
    size_int = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(words=words_list, size=size_int)


if __name__ == '__main__':
    main()





