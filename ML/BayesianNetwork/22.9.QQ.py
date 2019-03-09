# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import pandas as pd
import jieba
import jieba.posseg


# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_stopword():
    f_stop = open('stopword.txt')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw


def clean_info(info):
    replace_str = (('\n', ''), ('\r', ''), (',', '，'), ('[表情]', ''))
    for rs in replace_str:
        info = info.replace(rs[0], rs[1])

    at_pattern = re.compile(r'(@.* )')
    at = re.findall(pattern=at_pattern, string=info)
    for a in at:
        info = info.replace(a, '')
    idx = info.find('@')
    if idx != -1:
        info = info[:idx]
    return info


def regularize_data(file_name):
    time_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{1,2}:\d{1,2}:\d{1,2}')
    qq_pattern1 = re.compile(r'([1-9]\d{4,15})')    # QQ号最小是10000
    qq_pattern2 = re.compile(r'(\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*)')
    f = open(file_name, encoding='utf-8')
    f_output = open('QQ_chat.csv', mode='w', encoding='utf-8')
    f_output.write('QQ,Time,Info\n')
    qq = chat_time = info = ''
    for line in f:
        line = line.strip()
        if line:
            t = re.findall(pattern=time_pattern, string=line)
            qq1 = re.findall(pattern=qq_pattern1, string=line)
            qq2 = re.findall(pattern=qq_pattern2, string=line)
            if (len(t) >= 1) and ((len(qq1) >= 1) or (len(qq2) >= 1)):
                if info:
                    info = clean_info(info)
                    if info:
                        info = '%s,%s,%s\n' % (qq, chat_time, info)
                        f_output.write(info)
                        info = ''
                if len(qq1) >= 1:
                    qq = qq1[0]
                else:
                    qq = qq2[0][0]
                chat_time = t[0]
            else:
                info += line
    f.close()
    f_output.close()


def load_stopwords():
    stopwords = set()
    f = open('./stopword.txt', encoding='GBK')
    for w in f:
        stopwords.add(w.strip())
    f.close()
    return stopwords


def segment():
    stopwords = load_stopwords()
    data = pd.read_csv('QQ_chat.csv', header=0, encoding='utf-8')
    for i, info in enumerate(data['Info']):
        info_words = []
        for word, pos in jieba.posseg.cut(info):
            if pos in ['n', 'nr', 'ns', 'nt', 'nz', 's', 't', 'v', 'vd', 'vn', 'z', 'a', 'ad', 'an', 'f', 'i', 'j', 'Ng']:
                if word not in stopwords:
                    info_words.append(word)
        if info_words:
            data.iloc[i, 2] = ' '.join(info_words)
        else:
            data.iloc[i, 2] = np.nan
    data.dropna(axis=0, how='any', inplace=True)
    data.to_csv('QQ_chat_segment.csv', sep=',', header=True, index=False, encoding='utf-8')


def combine():
    data = pd.read_csv('QQ_chat_segment.csv', header=0, encoding='utf-8')
    data['QQ'] = pd.Categorical(data['QQ']).codes
    f_output = open('QQ_chat_result.csv', mode='w', encoding='utf-8')
    f_output.write('QQ,Info\n')
    for qq in data['QQ'].unique():
        info = ' '.join(data[data['QQ'] == qq]['Info'])
        str = '%s,%s\n' % (qq, info)
        f_output.write(str)
    f_output.close()


def export_perplexity1(corpus_tfidf, dictionary, corpus):
    lp1 = []
    lp2 = []
    topic_nums = np.arange(2, 51)
    for t in topic_nums:
        model = models.LdaModel(corpus_tfidf, num_topics=t, id2word=dictionary,
                                alpha=0.001, eta=0.02, minimum_probability=0,
                                update_every=1, chunksize=1000, passes=20)
        lp = model.log_perplexity(corpus)
        print('t = ', t, end=' ')
        print('lda.log_perplexity(corpus) = ', lp, end=' ')
        lp1.append(lp)

        lp = model.log_perplexity(corpus_tfidf)
        print('\t lda.log_perplexity(corpus_tfidf) = ', lp)
        lp2.append(lp)
    print(lp1)
    print(lp2)
    column_names = 'Topic', 'Perplexity_Corpus', 'Perplexity_TFIDF'
    perplexity_topic = pd.DataFrame(data=list(zip(topic_nums, lp1, lp2)), columns=column_names)
    perplexity_topic.to_csv('perplexity.csv', header=True, index=False)


def export_perplexity2(corpus_tfidf, dictionary, corpus):
    lp1 = []
    lp2 = []
    t = 20
    passes = np.arange(1, 20)
    for p in passes:
        model = models.LdaModel(corpus_tfidf, num_topics=t, id2word=dictionary,
                                alpha=0.001, eta=0.02, minimum_probability=0,
                                update_every=1, chunksize=100, passes=p)
        lp = model.log_perplexity(corpus)
        print('t = ', t, end=' ')
        print('lda.log_perplexity(corpus) = ', lp, end=' ')
        lp1.append(lp)

        lp = model.log_perplexity(corpus_tfidf)
        print('\t lda.log_perplexity(corpus_tfidf) = ', lp)
        lp2.append(lp)
    print(lp1)
    print(lp2)
    column_names = 'Passes', 'Perplexity_Corpus', 'Perplexity_TFIDF'
    perplexity_topic = pd.DataFrame(data=list(zip(passes, lp1, lp2)), columns=column_names)
    perplexity_topic.to_csv('perplexity2.csv', header=True, index=False)


def lda(export_perplexity=False):
    np.set_printoptions(linewidth=300)
    data = pd.read_csv('QQ_chat_result.csv', header=0, encoding='utf-8')
    texts = []
    for info in data['Info']:
        texts.append(info.split(' '))
    M = len(texts)
    print('文档数目：%d个' % M)
    # pprint(texts)

    print('正在建立词典 --')
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print('正在计算文本向量 --')
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('正在计算文档TF-IDF --')
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))
    print('LDA模型拟合推断 --')
    num_topics = 20
    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha=0.001, eta=0.02, minimum_probability=0,
                          update_every=1, chunksize=1000, passes=20)
    print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))
    if export_perplexity:
        export_perplexity1(corpus_tfidf, dictionary, corpus)
        # export_perplexity2(corpus_tfidf, dictionary, corpus)
    # # 所有文档的主题
    # doc_topic = [a for a in lda[corpus_tfidf]]
    # print 'Document-Topic:\n'
    # pprint(doc_topic)

    num_show_term = 7  # 每个主题显示几个词
    print('每个主题的词分布：')
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id, end=' ')
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        for t in term_id:
            print(dictionary.id2token[t], end=' ')
        print('\n概率：\t', term_distribute[:, 1])

    # 随机打印某10个文档的主题
    np.set_printoptions(linewidth=200, suppress=True)
    num_show_topic = 10  # 每个文档显示前几个主题
    print('10个用户的主题分布：')
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    idx = np.arange(M)
    np.random.shuffle(idx)
    idx = idx[:10]
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        # print topic_distribute
        topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
        print(('第%d个用户的前%d个主题：' % (i, num_show_topic)), topic_idx)
        print(topic_distribute[topic_idx])
    # 显示着10个文档的主题
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 7), facecolor='w')
    for i, k in enumerate(idx):
        ax = plt.subplot(5, 2, i + 1)
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        ax.stem(topic_distribute, linefmt='g-', markerfmt='ro')
        ax.set_xlim(-1, num_topics + 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("概率")
        ax.set_title("用户 {}".format(k))
        plt.grid(b=True, axis='both', ls=':', color='#606060')
    plt.xlabel("主题", fontsize=13)
    plt.suptitle('用户的主题分布', fontsize=15)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.show()

    # 计算各个主题的强度
    print('\n各个主题的强度:\n')
    topic_all = np.zeros(num_topics)
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    for i in np.arange(M):  # 遍历所有文档
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        topic_all += topic_distribute
    topic_all /= M  # 平均
    idx = topic_all.argsort()
    topic_sort = topic_all[idx]
    print(topic_sort)
    plt.figure(facecolor='w')
    plt.stem(topic_sort, linefmt='g-', markerfmt='ro')
    plt.xticks(np.arange(idx.size), idx)
    plt.xlabel("主题", fontsize=13)
    plt.ylabel("主题出现概率", fontsize=13)
    plt.title('主题强度', fontsize=15)
    plt.grid(b=True, axis='both', ls=':', color='#606060')
    plt.show()


def show_perplexity():
    data = pd.read_csv('Perplexity2.csv', header=0)
    print(data)
    columns = list(data.columns)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(data[columns[0]], data[columns[1]], 'ro-', lw=2, ms=6, label='Log Perplexity(Corpus)')
    # plt.plot(data[columns[0]], data[columns[2]], 'go--', lw=2, ms=6, label='Log Perplexity(TFIDF)')
    plt.legend(loc='lower left')
    plt.xlabel(columns[0], fontsize=16)
    plt.ylabel(columns[1], fontsize=16)
    plt.title('Perplexity', fontsize=18)
    plt.grid(b=True, axis='both', ls=':', color='#606060')
    plt.show()


if __name__ == '__main__':
    print('regularize_data')
    regularize_data('./chat.txt')
    print('segment')
    segment()
    print('combine')
    combine()
    print('lda')
    lda(export_perplexity=False)
    # show_perplexity()
