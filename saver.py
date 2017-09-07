from tqdm import tqdm
import os
def save_posts(dirpath,posts):
    print("save posts")
    for post in tqdm(posts):
        _filename = post["created_time"]
        
        filename = "".join(x for x in _filename if x.isalnum())
        path = os.path.join(dirpath,filename)+".txt"

        with open(path, 'w',encoding="utf-8") as f:
            f.write(post["message"])
