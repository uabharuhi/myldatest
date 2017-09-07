import facebook
import requests

class PostLoader():
    def __init__(self):
        self.id = "___"
        self.name = "unknown"
        self.posts = []
    def load_post(self,token):
        graph = facebook.GraphAPI(access_token = token)
        me_info = graph.get_object('me')
        self.name,self.id = me_info["name"],me_info["id"]

        posts = graph.get_connections(id = 'me', connection_name = 'posts')

        print("loading posts")
        while True:
            for post in posts['data']:
                if 'message' in post:
                    self.posts.append({"message":post['message'],"created_time":post['created_time']})
            try:
                posts = requests.get(posts['paging']['next']).json()
            except KeyError:
                break
            print("next page")

        print("load complete")
        return self.posts


            






