import scripts
import config
import util
import json
convert = None

def test_print_utf8_on_console():
    #from a file
    pass
def test_save_tokens2file():
    util.save_tokens_to_file("./data/1.txt","./temp/out.txt")

    ##can not write to console because comsole is not support utf-8..
    #for token in tokens :
        #print(token)

def test_stopwords():
    #l = util.word_segmentation("i love除了 you那邊好像 ＃ 那一個 ＄ 非但")
    #l = util.extract_stopwords(l)
    #util.print_tokens(l)
    pass

def test_vocab():
    pass
    #with open("./posts/20141220T0647290000.txt",encoding="utf-8")  as  f:
        #s = f.read()
        #_tokens = util.word_segmentation(s)
        #util.print_tokens(_tokens)
        #print('- - -- - - - - - -')
        #tokens = util.extract_stopwords(_tokens)
        #util.print_tokens(tokens)

def test_tfmatrix():
    print(' test_tfmatrix')
    with open(config.tfmatrix_path,"r",encoding="utf-8") as f:
        obj = json.loads(f.read())
        table,tf = obj["lookup"],obj["tf_matrix"]
    with open("./posts/20141220T0647290000.txt","r",encoding="utf-8") as f:
        words = util.extract_stopwords(util.word_segmentation(f.read()))
        cnt = [0]*len(tf[0])
        for word in words :
            cnt[table[word]]+=1
        assert cnt==tf[0]
        #assert cnt==tf[1]



        
import argparse


#run.py b  [-p <datapath>]] 
#       r  [-p <tfpath>] t <num_topic>  i <iter_num> d <display_num>]   
if __name__== "__main__": 
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(help='sub-command help',dest='parser_name')

    build_parser =   subparsers .add_parser('build')
    build_parser.add_argument('-p',"--datapath",default=config.data_path)
    build_parser.add_argument('-c',"--convert",action='store_true')

    run_parser =  subparsers .add_parser('run')
    run_parser.add_argument('-p','--tfpath',default=config.tfmatrix_path)
    run_parser.add_argument('-rp','--result_path',default=config.result_path)
    run_parser.add_argument('-c',"--convert",action='store_true')
    run_parser.add_argument('t',type=int)
    run_parser.add_argument('i',type=int)
    run_parser.add_argument('d',type=int)

  
    args = parser.parse_args()

    
    convert =  False
  
    #有提供-c參數代表要繁體轉簡體
    if args.parser_name == 'build':
        #print(args.datapath)
        scripts.build(data_path=args.datapath)
    elif args.parser_name == 'run':
        scripts.run(args.tfpath,args.result_path,args.t,args.i,args.d)
        



    #scripts.build()
    #scripts.run(topic_num=2,iter_num=1000,display_n=5)
    


#import sys

#seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
#openCC = OpenCC('t2s')  # convert from Simplified Chinese to Traditional Chinese
# can also set conversion by calling set_conversion
# openCC.set_conversion('s2tw')
#to_convert = '我們有清完標籤的語料了，第二件事就是要把語料中每個句子，進一步拆解成一個一個詞，這個步驟稱為「斷詞」。中文斷詞的工具比比皆是，這裏我採用的是 jieba，儘管它在繁體中文的斷詞上還是有些不如CKIP，但他實在太簡單、太方便、太好調用了，足以彌補這一點小缺憾：'
#converted = openCC.convert(to_convert)
#seg_list = jieba.cut(converted, cut_all=False)
#for s in seg_list:
#   sys.stdout.buffer.write(("[%s] "%(s)).encode('utf-8'))
#print("Default Mode: " + "/ ".join(seg_list)) # 精确模式
