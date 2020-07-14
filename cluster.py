import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score, silhouette_score
import argparse


def read(conn, name):
    df = pd.read_sql(f'SELECT vec, label_ids FROM {name} JOIN Files ON {name}.file_id == Files.file_id', conn)
    vectors = []
    for v in df['vec']:
        vectors.append(list(map(float, v.split(','))))
    labels = []
    for lbls in df['label_ids']:
        labels.append(list(map(int, lbls.split(','))))

    vectors_mult = []
    labels_mult = []
    for v, lbls in zip(vectors, labels):
        for l in lbls:
            vectors_mult.append(v)
            labels_mult.append(l)

    return np.asarray(vectors_mult), np.asarray(labels_mult)


def cluster(vec, n_clusters):
    return KMeans(n_clusters, random_state=42).fit(vec).labels_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mouse', 'trecgen', '20ng'])
    args = parser.parse_args()
   
    conn = sqlite3.connect(f'data/{args.dataset}.sqlite')
    vector_names = ['word2vec', 'pv_dm', 'pv_dbow']
 
    for name in vector_names:
        v, true_labels = read(conn, name)
        cluster_labels = cluster(v, len(set(true_labels)))
        print('{}: {:.4f}, {:.4f}'.format(
            name,
            adjusted_mutual_info_score(cluster_labels, true_labels),
            silhouette_score(v, cluster_labels)
        ))
