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
        with open(config.vocab_path,"w",encoding="utf-8") as f:
            json.dump(list(vocab),f)
    return vocab

def create_tf_matrix(dump=True):
    vocab = util.load_vocab()
    word_matrix = util.get_document_words("./posts/*.txt")
    id_matrix,id_table = util.doc_tokens2id(word_matrix,vocab)
    tf_matrix = util.tf_matrix(id_matrix,len(vocab))
    if dump:
        with open(config.tfmatrix_path,"w",encoding="utf-8") as f:
            obj = {"lookup":id_table,"tf_matrix":tf_matrix}
            json.dump(obj,f)
    return tf_matrix



def build(load=False):
    if load:
        load_my_posts()
    create_vocab(data_path=config.data_path)
    create_tf_matrix()

def run():
    with open(config.tfmatrix_path,"r",encoding="utf-8") as f:
        obj = json.loads(f.read())
        w2id_table,tf = obj["lookup"],obj["tf_matrix"]
    print(len(w2id_table))
    

    id2word_table = [None]*len(obj["lookup"])
    for word in list(w2id_table.keys()):
        id2word_table[w2id_table[word]] = word


    model = lda.LDA(n_topics=10, n_iter=100)
    model.fit(np.array(tf))
    topic_word = model.topic_word_


    n_top_words = 8


    res = []
    print("get result ...")
    for i, topic_dist in enumerate(topic_word):
        top_n_idx = np.argsort(topic_dist)[:-(n_top_words+1):-1]
        top_n_prob  = topic_dist[top_n_idx]
        topic_words = np.array(id2word_table)[top_n_idx]
        res.append([   topic_words.tolist(),   top_n_prob.tolist() ])
    print("save result to %s"%(config.result_path))
    util.show_result(res)
    with open(config.result_path,"w",encoding="utf-8") as f:
        json.dump(res,f)

