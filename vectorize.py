import sqlite3

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.corpus import stopwords
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA


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
    tagged_data = []
    for text_id, text in zip(ids, data):
        tagged_data.append(TaggedDocument(words=text, tags=[text_id]))

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
    tf = CountVectorizer().fit_transform(corpus)
    return LatentDirichletAllocation(
        n_components=100, random_state=42, n_jobs=4).fit_transform(tf).toarray()


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
    args = parser.parse_args()

    conn = sqlite3.connect('data/mouse.sqlite')
    stops = set(stopwords.words('english'))

    files = pd.read_sql('SELECT * FROM Files', conn)

    labels = [list(map(int, ids.split(','))) for ids in files['label_ids']]
    print(len(labels))
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
        print(vectors.shape)
    else:
        assert False, '{} is not implemented'.format(args.type)

    #vectors = PCA(n_components=30).fit_transform(vectors)

    emb = TSNE(random_state=42).fit_transform(vectors)
    print(emb.shape)

    for i in range(len(emb)):
        plt.plot(emb[i][0], emb[i][1], lbl2color(labels[i][0]), marker='')
        for lbl in labels[i]:
            plt.text(emb[i][0], emb[i][1], str(lbl), color=lbl2color(lbl), fontsize=12)
    plt.axis('off')
    plt.show()
