#! /usr/bin/env python
# coding: utf-8


"""
处理中文分词

author: lcj22
date: 2018-05-14

"""


import pandas as pd
import jieba as jb


from my_configs import Public, ProcessingWord

pub = Public()
pw = ProcessingWord()


def proessing_word(df, col, new_col="word", save_filename=None, stop_words=None, length=5):

    if stop_words is None:
    #     stop_words = [
    # u' ',   u"!",  u'"',  u"#",  u"$",  u"%",  u"&",  u"'",  u"(",  u")",  u"*",
    # u"+",  u",",  u"-",  u"--",  u".",  u"..",  u"...",  u"......",  u"...................",  u"./",  u".一",  u".数",
    # u".日",  u"/",  u"//",  u"0",  u"1",  u"2",  u"3",  u"4",  u"5",  u"6",  u"7",  u"8",  u"9",  u":",  u"://",  u"::",
    # u";",  u"<",  u"=",  u">",  u">>",  u"?",  u"@",  u"A",  u"Lex",  u"[",  u"\\",  u"]",  u"^",  u"_",  u"`",  u"exp",
    # u"sub",  u"sup",  u"|",  u"}",  u"~",  u"~~~~",  u"·",  u"×",  u"×××",  u"Δ",  u"Ψ",  u"γ",  u"μ",  u"φ",
    # u"φ．",  u"В",  u"—",  u"——",  u"———",  u"‘",  u"’",  u"’‘",  u"“",  u"”",  u"”，",  u"…",  u"……",
    # u"…………………………………………………③",  u"′∈",  u"′｜",  u"℃",  u"Ⅲ",  u"↑",  u"→",  u"∈［",
    # u"∪φ∈",  u"≈",  u"①",  u"②",  u"②ｃ",  u"③",  u"③］",  u"④",  u"⑤",  u"⑥",  u"⑦",  u"⑧",  u"⑨",
    # u"⑩",  u"──",  u"■",  u"▲",  u"　",  u"、",  u"。",  u"〈",  u"〉",  u"《",  u"》",  u"》），",  u"」",  u"『",
    # u"』",  u"【",  u"】",  u"〔",  u"〕",  u"〕〔",  u"㈧",  u"一",  u"一.",  u"一一", u'，'
    #                     ]
    #     stop_words = [
    #         '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '   ', '[', '{', ']', '}',
    #         '\\', '|',
    #         ';', ':', "'", '"', ',', '<', '.', '>', '/', '?', '·', '~', '！', '@', '#', '￥', '%', '……', '&', '*',
    #         '（', '）', '-', '——', '=', '+', '【', '｛', '】', '｝', '、', '|', '；', '：', '‘', '’', '“', '”',
    #         '，', '《', '。', '》', '、', '？'
    #     ]
        stop_words = [
            u"｛", u"，", u"｝", u"……", u"!", u"#", u'"',  u"%",  u"$",  u"'",  u"&",  u")",  u"(",  u"+",  u"*",
            u"-",  u",",  u"/",  u".",  u"￥",  u"》",  u";",  u":",  u"=",  u"<",  u"?",  u">",  u"@",  u"——",  
            u"。",  u"：",  u"；",  u"？",  u"！",  u"（",  u"）",  u"[",  u"]",  u"\\",  u"_",  u"^",  u"`",  u"   ",
            u"【",  u"】",  u"”",  u"“",  u"’",  u"《",  u"·",  u"‘",  u"、",  u"{",  u"}",  u"|",  u"~",  u" ",
            u""
        ]

    print u"共有：{} 个停用词".format(len(stop_words))

    drop_na_df = df.dropna()
    filter_df = drop_na_df[drop_na_df[col].str.len() >= length]
    duplicated_df = filter_df.drop_duplicates()
    duplicated_df.index = range(duplicated_df.shape[0])

    print u"共删掉了 {} 行空值和句子长度小于：{} 的行数据".format(df.shape[0] - filter_df.shape[0], length)

    all_words_list = []
    for i_int in filter_df.index:
        tmp_split_words_list = list(jb.cut(filter_df[col][i_int]))
        tmp_split_words_list = [val_str.encode("utf-8").decode("utf-8") for val_str in tmp_split_words_list]
        words_list = []
        for split_word_str in tmp_split_words_list:
            if split_word_str not in stop_words:
                words_list.append(split_word_str)
        all_words_list.extend(words_list)

    df = pd.DataFrame()
    df[new_col] = all_words_list
    len_more_than_0_df = df[df[new_col].str.len() >= 1]

    if save_filename:
        len_more_than_0_df.to_csv(save_filename, index=False, encoding="utf-8")

    return df


def main():
    filename_str = u"input_datas/datas_cn.csv"
    save_filename_str = u"output_datas/processed_datas_cn.csv"
    df = pd.read_csv(filename_str)
    proessing_word(df, col="REMARK", save_filename=save_filename_str)


if __name__ == '__main__':
    main()