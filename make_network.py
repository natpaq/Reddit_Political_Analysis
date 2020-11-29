import networkx as nx
import json


def network(data):
    print('hi')
    print(data.keys())


if __name__ == '__main__':
    with open('./user_data/biden_user_subreddit_freq.json') as json_file:
        data = json.load(json_file)
    network(data)
