import sys
from opencc import OpenCC 
import jieba
import config
import json
from tqdm import tqdm
import glob

stopwords =  []
openCC = OpenCC('t2s')


def dump_json_utf8(obj,path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False)   

def load_json_utf8(path):
    obj = None
    with open(path,"r",encoding="utf-8") as f:
        s = f.read()
        obj  =  json.loads(s,encoding="utf-8")
    return obj

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

def tokens_without_stopwords_from_file(path):
    with open(path,"r",encoding="utf-8") as f:
        s = f.read()
        return tokenized_without_stopwords(s)


def tokenized_without_stopwords(s):
    l= word_segmentation(s)
    return extract_stopwords(l)

def word_segmentation(s):
   # convert from Simplified Chinese to Traditional Chinese
    converted = openCC.convert(s)
    seg_list = jieba.cut(converted, cut_all=False)
    return list(seg_list)

def extract_stopwords(word_list):
    l = [ word for word in word_list if (not word.isspace()) and (word not in  stopwords) ]
    return l

def load_vocab():
    l = load_json_utf8(config.vocab_path)
    return l

def get_document_words(path):
    doc_tokens = []
    for filename in tqdm(glob.glob(path)):
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

def save_tokens_to_file(path,save_to):
        tokens =  tokens_from_file(path)
        with open(save_to,"w",encoding="utf-8") as f:
            for w in tokens:
                f.write(w+"\n")

def tokens_from_file(path):
    with open(path,"r",encoding="utf-8") as f:
        s = f.read()
        l = tokenized_without_stopwords(s)
    return  l


load_stopwords(config.stopwordpath)