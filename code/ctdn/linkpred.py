import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def load_node_embeddings(path_to_w2v):
    '''
    load the saved word2vec representation
    :return:
    '''
    with open(path_to_w2v, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def load_edges(save_path):
    with open(save_path + 'pos_edges', 'rb') as f:
        pos_edges = pickle.load(f)
    with open(save_path + 'neg_edges', 'rb') as f:
        neg_edges = pickle.load(f)

    return pos_edges, neg_edges


def operator(u, v, op='mean'):
    if op=='mean':
        return (u + v)/2.0
    elif op=='l1':
        return np.abs(u - v)
    elif op == 'l2':
        return np.abs(u - v)**2
    elif op == 'hadamard':
        return np.multiply(u, v)
    else:
        return None


def get_dataset_from_embedding(embeddings, pos_edges, neg_edges, op='mean'):
    '''
    op can take values from 'mean', 'l1', 'l2', 'hadamard'
    '''
    y = []
    X = []

    # process positive links
    for u, v, prop in pos_edges:
        # get node representation and average them
        u_enc = embeddings.get(u)
        v_enc = embeddings.get(v)

        if (u_enc is None) or (v_enc is None):
            continue

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc)/2.0

        X.append(datapoint)
        y.append(0.0)

    # process negative links
    for u, v, prop in neg_edges:
        # get node representation and average them
        u_enc = embeddings.get(u)
        v_enc = embeddings.get(v)

        if (u_enc is None) and (v_enc is not None):
            u_enc = v_enc
        if (v_enc is None) and (u_enc is not None):
            v_enc = u_enc
        if (u_enc is None) and (v_enc is None):
            continue

        datapoint = operator(u_enc, v_enc, op=op)  # (u_enc + v_enc) / 2.0

        X.append(datapoint)
        y.append(1.0)

    dataset = np.array(X), np.array(y)
    return dataset


def main():
    embeddings_path = '../../network_data/ia-contact/ia-contact.w2v.pkl'
    embeddings = load_node_embeddings(embeddings_path)

    edges_save_basepath = '../../network_data/ia-contact/'
    pos_edges, neg_edges = load_edges(edges_save_basepath)

    X, y = get_dataset_from_embedding(embeddings, pos_edges, neg_edges)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    logReg = LogisticRegression(solver='lbfgs')
    logReg.fit(X_train, y_train)

    y_pred = logReg.predict(X_test)
    y_score = logReg.predict_proba(X_test)
    print(y_score.shape)

    print('Link prediction accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Link prediction roc:', metrics.roc_auc_score(y_true=y_test, y_score=y_score[:, 1]))


if __name__ == '__main__':
    main()
