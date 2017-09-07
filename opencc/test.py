from opencc import OpenCC 
import jieba
import sys

openCC = OpenCC('t2s')  # convert from Simplified Chinese to Traditional Chinese
# can also set conversion by calling set_conversion
# openCC.set_conversion('s2tw')
to_convert = 'そして 次の曲が始まるのです補這一點小缺憾：'
converted = openCC.convert(to_convert)
seg_list = jieba.cut(converted, cut_all=False)
for s in seg_list:
	sys.stdout.buffer.write(("[%s] "%(s)).encode('utf-8'))
#print("Default Mode: " + "/ ".join(seg_list)) # 精确模式