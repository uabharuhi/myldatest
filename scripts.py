from   postloader  import PostLoader
import config
import saver
import util
import glob
from tqdm import tqdm
import json
import lda
import numpy as np

def load_my_posts():
    loader = PostLoader()
    posts = loader.load_post(config.token)
    saver.save_posts(config.save_post_to,posts)

def  create_vocab(data_path,dump=True):
    vocab = set()
    for filename in tqdm(glob.glob(data_path)):
        with open(filename,"r",encoding="utf-8") as f:
            text = f.read()
            _tokens = util.word_segmentation(text)
            tokens = util.extract_stopwords(_tokens)
            vocab = vocab | set(tokens)
    if dump:
        print("dump vocab to file %s"%(config.vocab_path))
        util.dump_json_utf8(list(vocab),config.vocab_path)
    return vocab

def create_tf_matrix(dump=True):
    print('create term frequncy matrix')
    vocab = util.load_vocab()
    print("vocab size",len(vocab))
    word_matrix = util.get_document_words(config.data_path)
    id_matrix,id_table = util.doc_tokens2id(word_matrix,vocab)
    tf_matrix = util.tf_matrix(id_matrix,len(vocab))
    if dump:
        obj = {"lookup":id_table,"tf_matrix":tf_matrix}
        util.dump_json_utf8(obj,config.tfmatrix_path)
    return tf_matrix


def build(load=False):
    print('buiding ....')
    if load:
        load_my_posts()
    create_vocab(data_path=config.data_path)
    create_tf_matrix()

def run(topic_num,iter_num,display_n):
    print("running...")
    obj = util.load_json_utf8(config.tfmatrix_path)
    w2id_table,tf = obj["lookup"],obj["tf_matrix"]
    print(len(w2id_table))
    
    id2word_table = [None]*len(obj["lookup"])
    for word in list(w2id_table.keys()):
        id2word_table[w2id_table[word]] = word


    model = lda.LDA(n_topics=topic_num, n_iter=iter_num)
    model.fit(np.array(tf))
    topic_word = model.topic_word_


    n_top_words = display_n


    res = []
    print("get result ...")
    for i, topic_dist in enumerate(topic_word):
        top_n_idx = np.argsort(topic_dist)[:-(n_top_words+1):-1]
        top_n_prob  = topic_dist[top_n_idx]
        topic_words = np.array(id2word_table)[top_n_idx]
        res.append([   topic_words.tolist(),   top_n_prob.tolist() ])
    print("save result to %s"%(config.result_path))

    util.show_result(res)

    util.dump_json_utf8(res,config.result_path)
