'''
__author__ : 'Shubhranshu Shekhar'
This file generates the time honoring random walks given a graph
'''
import gensim
import numpy as np
import pickle


def learn_node_representation(time_walks, size=128):
    documents = []
    for walk in time_walks:
        documents.append([str(w) for w in walk])

    # w2v model
    model = gensim.models.Word2Vec(documents, size=size, window=10, min_count=1, workers=4, negative=10)
    model.train(documents, total_examples=len(documents), epochs=200)

    # save the vectors into a dict and then pickle it
    node_representation = {}
    for key, val in model.wv.vocab.items():
        node_representation[int(key)] = model.wv[key]

    return node_representation


def main():
    path = '../../network_data/ia-contact/ia-contact.time.walks'
    w2v_embedding_path = '../../network_data/ia-contact/ia-contact.w2v.pkl'

    with open(path, 'rb') as f:
        time_walks = pickle.load(f)

    node_representation = learn_node_representation(time_walks)

    with open(w2v_embedding_path, 'wb') as f:
        pickle.dump(node_representation, f)


if __name__ == '__main__':
    main()
