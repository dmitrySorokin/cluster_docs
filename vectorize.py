import sqlite3

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score, silhouette_score

from nltk.corpus import stopwords
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
import json
from pprint import pprint


def w2v(data, use_tfidf=False, use_idf=False, model_path='data/word2vec_sg0'):
    model = Word2Vec(
        data, 
        size=100, 
        window=5, 
        min_count=5, 
        workers=4, 
        iter=20, 
        sg=0
    )

    #if os.path.isfile(model_path):
    #    model = Word2Vec.load(model_path)
    #else:
    #    model = Word2Vec(data, size=100, window=5, min_count=5, workers=4, iter=15, sg=0)
    #    model.save(model_path)
    #    print('Model Saved:', model_path)

    if use_tfidf or use_idf:
        tfidf = TfidfVectorizer(tokenizer=lambda x: x, stop_words=[], lowercase=False)
        matrix = tfidf.fit_transform(data).toarray()
        feature_names = tfidf.get_feature_names()
        tfidf_score = {nm: val for nm, val in zip(feature_names, matrix.T)}
        idf_score = {nm: val for nm, val in zip(feature_names, tfidf.idf_)}

    result = []
    for text_id, text in enumerate(data):
        word_vectors = []
        for word in text:
            if word in model.wv.vocab:
                word_vectors.append(model.wv[word])
                if use_idf:
                    word_vectors[-1] = word_vectors[-1] * idf_score.get(word)
                if use_tfidf:
                    word_vectors[-1] = word_vectors[-1] * tfidf_score.get(word)[text_id]
        if len(word_vectors) > 0:
            result.append(np.mean(word_vectors, axis=0))
        else:
            result.append([0] * 100)
    return np.asarray(result)


def d2v(data, ids, dm, model_path):
    tagged_data = [
        TaggedDocument(words=text, tags=[text_id])
        for text, text_id in zip(data, ids)
    ]

    model = Doc2Vec(
        tagged_data, 
        vector_size=100, 
        window=5, 
        min_count=5, 
        workers=4, 
        epochs=20, 
        dm=dm)
    #if os.path.isfile(model_path):
    #    model = Doc2Vec.load(model_path)
    #else:
    #    model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=5, workers=4, epochs=15, dm=dm)
    #    model.save(model_path)
    #    print("Model Saved:", model_path)


    result = []
    for doc in tagged_data:
        result.append(*model[doc.tags])
    return np.asarray(result)


def lsa(corpus):
    tfidf = TfidfVectorizer().fit_transform(corpus)#.toarray()
    return TruncatedSVD(n_components=100).fit_transform(tfidf)


def lda(corpus):
    tf = CountVectorizer().fit_transform(corpus)#.toarray()
    return LatentDirichletAllocation(
            n_components=100,
            random_state=42,
            n_jobs=-1,
            verbose=1).fit_transform(tf)


def rdf(corpus, model_path='data/dict.jsonld', verbose=0):
    with open(model_path, encoding="utf8") as f:
        data = json.load(f)
    if verbose:
        pprint(data)
    w2i, i2w, ids = {}, {}, []
    for d in data['@graph']:
        if d['@type'] != 'skos:Concept':
            continue
        i = d['@id']
        ids.append(i)
        words = []
        for key in ['skos:prefLabel', 'skos:altLabel', 'skos:hiddenLabel']:
            pairs = d.get(key, {}).items()
            for lang, labels in pairs:
                if isinstance(labels, list):
                    words.extend(labels)
                else:
                    words.append(labels)
        # print(words)
        for w in words:
            w2i.setdefault(w, []).append(len(ids) - 1)
            i2w.setdefault(len(ids) - 1, []).append(w)
    vectors = np.zeros((len(corpus), len(ids)))
    for index, row in enumerate(corpus):
        tokens = row.split()
        for k in range(1, 3):
            for j in range(0, len(tokens), k):
                t = ' '.join(tokens[j:j + k])
                i = w2i.get(t, None)
                if i is not None:
                    vectors[index, i] += 1
    vectors /= vectors.sum(axis=1, keepdims=True)
    vectors = np.nan_to_num(vectors, nan=0)
    return vectors


def topic_net(files):
    data_path = 'data/topic_net.csv'
    df = pd.read_csv(data_path, usecols=['file_paths', 'vec'],
                     converters={'vec': lambda x: json.loads(x)})
    dim = len(df['vec'][0])
    print(f'vectors: {len(df)}, dim: {dim}')
    fp2i = {x: i for i, x in enumerate(df['file_paths'])}
    vectors = []
    for i, row in files.iterrows():
        fp = row['file_path']
        i2 = fp2i.get(fp, None)
        if i2 is not None:
            vectors.append(df['vec'][i2])
        else:
            print(fp)
            vectors.append([0 for _ in range(dim)])
    vectors = np.array(vectors)
    return vectors


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
    parser.add_argument('--type', choices=[
        'word2vec', 'word2vec_idf', 'word2vec_tfidf','pv_dm', 'pv_dbow', 'lsa', 'lda', 'rdf', 'topic_net'])
    parser.add_argument('--labels', choices=['db', 'cluster'])
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--dataset', choices=['mouse', 'trecgen', '20ng'])
    args = parser.parse_args()

    conn = sqlite3.connect(f'data/{args.dataset}.sqlite')
    stops = set(stopwords.words('english'))
    stops.add('http')

    files = pd.read_sql('SELECT * FROM Files', conn)

    texts = [list(filter(lambda w: w not in stops, text.split())) for text in files['text']]
    text_ids = list(map(int, files['file_id']))

    if args.type == 'word2vec':
        vectors = w2v(texts, model_path=f'data/word2vec_sg0_{args.dataset}')
    elif args.type == 'word2vec_idf':
        vectors = w2v(texts, use_idf=True)
    elif args.type == 'word2vec_tfidf':
        vectors = w2v(texts, use_tfidf=True)
    elif args.type == 'pv_dm':
        vectors = d2v(texts, text_ids, dm=1, model_path=f'data/pvdm_{args.dataset}')
    elif args.type == 'pv_dbow':
        vectors = d2v(texts, text_ids, dm=0, model_path=f'data/pvdbow_{args.dataset}')
    elif args.type == 'lsa':
        vectors = lsa(files['text'])
    elif args.type == 'lda':
        vectors = lda(files['text'])
    elif args.type == 'rdf':
        vectors = rdf(files['text'])
    elif args.type == 'topic_net':
        vectors = topic_net(files)
    else:
        assert False, '{} is not implemented'.format(args.type)

    #for i, v in enumerate(vectors):
    #    vectors[i] = v / np.linalg.norm(v)

    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS {args.type}')
    cursor.execute(f'CREATE TABLE {args.type}(file_id INTEGER NOT NULL PRIMARY KEY, vec TEXT NOT NULL)')
    for text_id, v in zip(text_ids, vectors):
        vec = ','.join(map(str, v))
        cursor.execute(f'INSERT INTO {args.type} VALUES ({text_id}, "{vec}")')
    conn.commit()

    if args.labels == 'db':
        labels = [list(map(int, ids.split(','))) for ids in files['label_ids']]
    elif args.labels == 'cluster':
        labels = cluster(vectors, n_clusters=27)
    else:
        assert False, '{} is not implemented'.format(args.labels)

    if args.save:
        with open('{}.csv'.format(args.save), 'w') as out:
            file_path = list(files['file_path'])
            out.write('file_id\tfile_path\tlabel\n')
            for i in np.argsort(labels):
                out.write('{}\t{}\t{}\n'.format(text_ids[i], file_path[i], labels[i]))

    if args.plot:
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

    print(vectors.shape)
    labels = np.asarray(labels).ravel()
    print(labels.shape)
    score = silhouette_score(vectors, labels)
    print('silhouette_score', score)
