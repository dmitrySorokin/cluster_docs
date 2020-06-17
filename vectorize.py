import sqlite3

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

from nltk.corpus import stopwords
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA

from viz import contingency_matrix as cmat


def w2v(data, model_path='data/word2vec_sg0'):
    if os.path.isfile(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(data, size=100, window=5, min_count=5, workers=4, iter=15, sg=0)
        model.save(model_path)
        print('Model Saved:', model_path)

    result = []
    for text in data:
        word_vectors = []
        for word in text:
            if word in model.wv.vocab:
                word_vectors.append(model.wv[word])
        if len(word_vectors) > 0:
            result.append(np.mean(word_vectors, axis=0))
        else:
            result.append([0] * 100)
    return np.asarray(result)


def d2v(data, ids, model_path='data/doc2vec_dm1'):
    tagged_data = [
        TaggedDocument(words=text, tags=[text_id])
        for text, text_id in zip(data, ids)
    ]

    if os.path.isfile(model_path):
        model = Doc2Vec.load(model_path)
    else:
        model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=5, workers=4, epochs=15, dm=1)
        model.save(model_path)
        print("Model Saved:", model_path)


    result = []
    for doc in tagged_data:
        result.append(*model[doc.tags])
    return np.asarray(result)


def lsa(corpus):
    tfidf = TfidfVectorizer().fit_transform(corpus).toarray()
    return TruncatedSVD(n_components=100).fit_transform(tfidf)


def lda(corpus):
    tf = CountVectorizer().fit_transform(corpus).toarray()
    return LatentDirichletAllocation(
        n_components=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    ).fit_transform(tf)


def cluster(vec, n_clusters):
    return KMeans(n_clusters, random_state=42).fit(vec).labels_


def lbl2color(l):
    colors = [
        "#cc4767", "#6f312a", "#d59081", "#d14530", "#d27f35",
        "#887139", "#d2b64b", "#c7df48", "#c0d296", "#5c8f37",
        "#364d26", "#70d757", "#60db9e", "#4c8f76", "#75d4d5",
        "#6a93c1", "#616ed0", "#46316c", "#8842c4", "#bc87d0"
    ]
    return colors[l % len(colors)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['word2vec', 'doc2vec', 'lsa', 'lda'])
    parser.add_argument('--labels', choices=['db', 'cluster'])
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--cmat', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect('data/mouse.sqlite')
    stops = set(stopwords.words('english'))

    files = pd.read_sql('SELECT * FROM Files', conn)

    texts = [list(filter(lambda w: w not in stops, text.split())) for text in files['text']]
    text_ids = list(map(int, files['file_id']))

    if args.type == 'word2vec':
        vectors = w2v(texts)
    elif args.type == 'doc2vec':
        vectors = d2v(texts, text_ids)
    elif args.type == 'lsa':
        vectors = lsa(files['text'])
    elif args.type == 'lda':
        vectors = lda(files['text'])
    else:
        assert False, '{} is not implemented'.format(args.type)

    if args.labels == 'db':
        labels = [list(map(int, ids.split(','))) for ids in files['label_ids']]
    elif args.labels == 'cluster':
        labels = cluster(vectors, n_clusters=9)
    else:
        assert False, '{} is not implemented'.format(args.labels)

    if args.save:
        with open('{}.csv'.format(args.save), 'w') as out:
            file_path = list(files['file_path'])
            out.write('file_id\tfile_path\tlabel\n')
            for i in np.argsort(labels):
                out.write('{}\t{}\t{}\n'.format(text_ids[i], file_path[i], labels[i]))

    #vectors = PCA(n_components=30).fit_transform(vectors)

    emb = TSNE(random_state=42).fit_transform(vectors)
    print(emb.shape)

    for i in range(len(emb)):
        plt.plot(emb[i][0], emb[i][1], marker='')
        if args.labels == 'db':
            for lbl in labels[i]:
                plt.text(emb[i][0], emb[i][1], str(lbl), color=lbl2color(lbl), fontsize=12)
        elif args.labels == 'cluster':
            plt.text(emb[i][0], emb[i][1], str(labels[i]), color=lbl2color(labels[i]), fontsize=12)

    plt.axis('off')
    plt.show()

    if args.cmat:
        db_labels = pd.read_sql("SELECT * FROM Labels", conn)
        i2l = dict(zip(db_labels['label_id'], db_labels['label_desc']))
        df = files
        y = [int(x.split(',')[0]) for x in df['label_ids']]  # FIXME only first
        y_pred = labels  # FIXME Only for --labels cluster
        y_labels = [i2l[x] for x in y]  # None
        df['labels'] = [','.join(i2l[int(l)] for l in x.split(','))
                        for x in df['label_ids']]
        del df['text']  # due to performance issues
        cmat(X=emb,
             y=y,
             y_pred=y_pred,
             df=df,
             tooltip_cols=['file_id',
                           'file_path',
                           'label_ids',
                           'labels'],
             y_labels=y_labels,
             y_pred_labels=None,
             cmap='tableau20',  # https://vega.github.io/vega/docs/schemes/
             filename='cm.html',
             sort=True)
