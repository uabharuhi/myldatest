import sys
from opencc import OpenCC 
import jieba
import config
import json
from tqdm import tqdm
import glob

stopwords =  []
def load_stopwords(path):
    global stopwords
    with open(path,"r",encoding="utf-8") as file:
        stopwords= file.read().splitlines()

def print_tokens(l):
    for s in l:
        print_utf8(s)
        print(" ")
def print_utf8(s):
    sys.stdout.buffer.write(s.encode('utf-8'))

def word_segmentation(s):
    openCC = OpenCC('t2s')  # convert from Simplified Chinese to Traditional Chinese
    converted = openCC.convert(s)
    seg_list = jieba.cut(converted, cut_all=False)
    return list(seg_list)

def extract_stopwords(word_list):
    l = [ word for word in word_list if (not word.isspace()) and (word not in  stopwords) ]
    return l

def load_vocab():
    l = None
    with open(config.vocab_path,"r",encoding="utf-8") as f:
        l = json.loads(f.read())
        #util.print_tokens(l)
    return l

def get_document_words(path):
    doc_tokens = []
    for filename in tqdm(glob.glob("./posts/*.txt")):
        with open(filename,"r",encoding="utf-8") as f:
            text = f.read()
            _tokens = word_segmentation(text)
            tokens =  extract_stopwords(_tokens)
            doc_tokens.append(tokens)
    return doc_tokens

def doc_tokens2id(doc_tokens,vocab):
    doc_ids = []
    id_dict  = {}
    for i,word in enumerate(vocab):
        id_dict[word] = i

    for words in doc_tokens:
        l = []
        for word in words:
            if word in id_dict:
                l.append(id_dict[word])
            else:
                print('doc_tokens2id--> word not in vocabulary set??? ')
        doc_ids.append(l)
    return doc_ids,id_dict

#                   word ids
def tf_matrix(doc_ids,V):
    matrix =[]
    for ids in  doc_ids:
        l = [0]*V
        for _id in ids:
            l[_id]+=1
        matrix.append(l)
    return matrix
            
               # array  of word list , # array  of prob list
def show_result(res):  
    for topic_i,info in enumerate(res):
        word_prob_tuple = zip(info[0],info[1])
        print("Topic %d"%(topic_i))
        for i,pair in enumerate(word_prob_tuple):
            print_utf8("[%s %.3f] "%(pair[0],pair[1]))
        print(" ")




load_stopwords(config.stopwordpath)