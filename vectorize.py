import sqlite3

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score, \
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, \
    completeness_score, fowlkes_mallows_score, \
    homogeneity_completeness_v_measure, homogeneity_score, \
    mutual_info_score, normalized_mutual_info_score, \
    silhouette_score, silhouette_samples, v_measure_score

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

from viz import contingency_matrix as cmat
from viz import metrics as metrics_viz
from viz import unsupervised_metrics as unsupervised_metrics_viz


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
            verbose=1).fit_transform(tf)


def test(corpus):
    np.random.seed(42)
    return np.random.rand(len(corpus), 2)


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
    parser.add_argument('--type', choices=['word2vec', 'doc2vec', 'lsa', 'lda',
                                           'rdf', 'test', 'topic_net'])
    parser.add_argument('--type2', choices=['word2vec', 'doc2vec', 'lsa', 'lda',
                                            'rdf', 'test', 'topic_net'],
                        default=None)
    parser.add_argument('--labels', choices=['db', 'cluster'])
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--cmat', action='store_true')
    parser.add_argument('--from_file', action='store_true')
    parser.add_argument('--metrics', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect('data/mouse.sqlite')
    stops = set(stopwords.words('english'))

    files = pd.read_sql('SELECT * FROM Files', conn)

    texts = [list(filter(lambda w: w not in stops, text.split())) for text in files['text']]
    text_ids = list(map(int, files['file_id']))

    vectors_path = os.path.join('data', args.type + '_vectors.npy')
    if args.from_file and os.path.exists(vectors_path):
        print(f'loading vectors from {vectors_path}')
        vectors = np.load(vectors_path)
    else:
        if args.type == 'word2vec':
            vectors = w2v(texts)
        elif args.type == 'doc2vec':
            vectors = d2v(texts, text_ids)
        elif args.type == 'lsa':
            vectors = lsa(files['text'])
        elif args.type == 'lda':
            vectors = lda(files['text'])
        elif args.type == 'test':
            vectors = test(files['text'])
        elif args.type == 'rdf':
            vectors = rdf(files['text'])
        elif args.type == 'topic_net':
            vectors = topic_net(files)
        else:
            assert False, '{} is not implemented'.format(args.type)
        if args.from_file:
            print(f'saving vectors to {vectors_path}')
            np.save(vectors_path, vectors)

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
        width = 600
        height = 600
        cmap = 'tableau20'  # https://vega.github.io/vega/docs/schemes/
        cm_sort = True
        sort_type = 'rc'
        inter_type = 'mat_leg'
        if args.type2 is None:
            db_labels = pd.read_sql("SELECT * FROM Labels", conn)
            i2l = dict(zip(db_labels['label_id'], db_labels['label_desc']))
            # expand files (one label per file)
            y = [int(label_id) for i, ids in enumerate(files['label_ids'])
                 for label_id in ids.split(',')]
            y_labels = [i2l[x] for x in y]  # None
            new_ids = [i for i, ids in enumerate(files['label_ids'])
                       for label_id in ids.split(',')]
            y_pred = [labels[i] for i in new_ids]
            # y_pred = labels  # FIXME Only for --labels cluster
            y_pred_labels = None
            X = None
            X_pred = np.array([emb[i] for i in new_ids])
            print(X_pred.shape)
            new_file_id = [files['file_id'][i] for i in new_ids]
            new_file_path = [files['file_path'][i] for i in new_ids]
            new_label_ids = [files['label_ids'][i] for i in new_ids]
            new_labels = [','.join(i2l[int(l)] for l in x.split(',')) for x in
                          new_label_ids]
            new_text = [files['text'][i] for i in new_ids]
            df = pd.DataFrame({
                'file_id': new_file_id,
                'file_path': new_file_path,
                'label_ids': new_label_ids,
                'labels': new_labels,
                'text': new_text})
            df['labels'] = [','.join(i2l[int(l)] for l in x.split(','))
                            for x in df['label_ids']]
            df['file_name'] = [os.path.basename(x) for x in df['file_path']]
            df['file_path'] = ['//' + x for x in df['file_path']]
            df['collection_id'] = y
            del df['text']  # due to performance issues
            tooltip_cols = ['file_id', 'file_name', 'file_path', 'label_ids',
                            'labels', 'collection_id']
            table_cols = ['file_id', 'file_name', 'label',
                          'label_id_pred', 'collection_id']
            href = 'file_path'
            cm_filename = f'cm_{args.type}.html'
        else:
            vectors2_path = os.path.join('data', args.type + '_vectors2.npy')
            if args.from_file and os.path.exists(vectors2_path):
                print(f'loading vectors2 from {vectors2_path}')
                vectors2 = np.load(vectors2_path)
            else:
                if args.type2 == 'word2vec':
                    vectors2 = w2v(texts)
                elif args.type2 == 'doc2vec':
                    vectors2 = d2v(texts, text_ids)
                elif args.type2 == 'lsa':
                    vectors2 = lsa(files['text'])
                elif args.type2 == 'lda':
                    vectors2 = lda(files['text'])
                elif args.type2 == 'test':
                    vectors2 = test(files['text'])
                elif args.type2 == 'rdf':
                    vectors2 = rdf(files['text'])
                elif args.type2 == 'topic_net':
                    vectors2 = topic_net(files)
                else:
                    assert False, '{} is not implemented'.format(args.type2)
                if args.from_file:
                    print(f'saving vectors2 to {vectors2_path}')
                    np.save(vectors2_path, vectors2)
            labels2 = cluster(vectors2, n_clusters=20)
            emb2 = TSNE(random_state=42).fit_transform(vectors2)
            print(emb2.shape)
            y = labels
            y_labels = None
            y_pred = labels2
            y_pred_labels = None
            X = emb
            X_pred = emb2
            df = files
            df['file_name'] = [os.path.basename(x) for x in df['file_path']]
            df['file_path'] = ['//' + x for x in df['file_path']]
            del df['text']  # due to performance issues
            tooltip_cols = ['file_id', 'file_name', 'file_path']
            table_cols = ['file_id', 'file_name', 'label_id', 'label_id_pred']
            href = 'file_path'
            cm_filename = f'cm_{args.type}_{args.type2}.html'
        cmat(y=y,
             y_pred=y_pred,
             df=df,
             X=X,
             X_pred=X_pred,
             tooltip_cols=tooltip_cols,
             table_cols=table_cols,
             href=href,
             width=width,
             height=height,
             y_labels=y_labels,
             y_pred_labels=y_pred_labels,
             cmap=cmap,
             filename=cm_filename,
             sort=cm_sort,
             sort_type=sort_type,
             inter_type=inter_type)
    if args.metrics:
        # supervised
        print('adjusted_mutual_info_score')  # -? to 1 -> 1 better
        print(adjusted_mutual_info_score(y, y_pred))
        print(adjusted_mutual_info_score(y_pred, y))
        print('adjusted_rand_score')  # 0 to 1 -> 1 better
        print(adjusted_rand_score(y, y_pred))
        print(adjusted_rand_score(y_pred, y))
        print('completeness_score')  # 0 to 1 -> 1 better
        print(completeness_score(y, y_pred))
        print(completeness_score(y_pred, y))
        print('fowlkes_mallows_score')  # 0 to 1 -> 1 better
        print(fowlkes_mallows_score(y, y_pred))
        print(fowlkes_mallows_score(y_pred, y))
        print('homogeneity_score')  # 0 to 1 -> 1 better
        print(homogeneity_score(y, y_pred))
        print(homogeneity_score(y_pred, y))
        print('normalized_mutual_info_score')  # 0 to 1 -> 1 better
        print(normalized_mutual_info_score(y, y_pred))
        print(normalized_mutual_info_score(y_pred, y))
        print('normalized_mutual_info_score')  # 0 to 1 -> 1 better
        print(mutual_info_score(y, y_pred))
        print(mutual_info_score(y_pred, y))
        print('v_measure_score')  # 0 to 1 -> 1 better
        print(v_measure_score(y, y_pred))
        print(v_measure_score(y_pred, y))
        # unsupervised
        if X is not None:
            print('calinski_harabasz_score')
            print(calinski_harabasz_score(X, y))  # higher -> better
            print('davies_bouldin_score')
            print(davies_bouldin_score(X, y))  # -> 0 better
            print('silhouette_score')
            print(silhouette_score(X, y))
            # -1 to 1 -1 for incorrect clustering
            # +1 for highly dense clustering.
            # Scores around zero indicate overlapping clusters.
        print('calinski_harabasz_score')
        print(calinski_harabasz_score(X_pred, y_pred))
        print('davies_bouldin_score')
        print(davies_bouldin_score(X_pred, y_pred))
        print('silhouette_score')
        print(silhouette_score(X_pred, y_pred))
        # supervized
        types = ['db', 'word2vec', 'doc2vec', 'lsa', 'lda', 'rdf', 'topic_net']
        paths = [os.path.join('data', x + '_vectors.npy') for x in types]
        metrics = [adjusted_mutual_info_score,
                   adjusted_rand_score, fowlkes_mallows_score,
                   completeness_score, homogeneity_score, v_measure_score,
                   normalized_mutual_info_score, mutual_info_score]
        M = np.zeros((len(metrics), len(types), len(types)))
        for i, (p, t) in enumerate(zip(paths, types)):
            if t == 'db':
                db_labels = pd.read_sql("SELECT * FROM Labels", conn)
                i2l = dict(zip(db_labels['label_id'], db_labels['label_desc']))
                # expand files (one label per file)
                y = [int(label_id) for i, ids in enumerate(files['label_ids'])
                     for label_id in ids.split(',')]
                new_ids = [i for i, ids in enumerate(files['label_ids'])
                           for label_id in ids.split(',')]
            else:
                if not os.path.exists(p):
                    continue
                vec = np.load(p)
                y = cluster(vec, n_clusters=20)
            for j, (p2, t2) in enumerate(zip(paths, types)):
                print(i, j, t, t2)
                if t2 == 'db':
                    db_labels = pd.read_sql("SELECT * FROM Labels", conn)
                    i2l = dict(
                        zip(db_labels['label_id'], db_labels['label_desc']))
                    # expand files (one label per file)
                    y_pred = [int(label_id) for i, ids in
                              enumerate(files['label_ids'])
                              for label_id in ids.split(',')]
                    new_ids = [i for i, ids in enumerate(files['label_ids'])
                               for label_id in ids.split(',')]
                    for k, m in enumerate(metrics):
                        if t != 'db':
                            M[k, i, j] = m([y[i] for i in new_ids], y_pred)
                        else:
                            M[k, i, j] = m(y, y_pred)
                else:
                    if not os.path.exists(p2):
                        continue
                    vec2 = np.load(p2)
                    y_pred = cluster(vec2, n_clusters=20)
                    if t == 'db':
                        y_pred = [y_pred[i] for i in new_ids]
                    for k, m in enumerate(metrics):
                        M[k, i, j] = m(y, y_pred)
        metrics_viz(M, [m.__name__ for m in metrics], types)
        # unsupervized
        types = ['word2vec', 'doc2vec', 'lsa', 'lda', 'rdf', 'topic_net']
        paths = [os.path.join('data', x + '_vectors.npy') for x in types]
        metrics = [calinski_harabasz_score, davies_bouldin_score,
                   silhouette_score]
        M = np.zeros((len(metrics), len(types)))
        for i, m in enumerate(metrics):
            for j, (p, t) in enumerate(zip(paths, types)):
                if not os.path.exists(p):
                    continue
                vec = np.load(p)
                y = cluster(vec, n_clusters=20)
                M[i, j] = m(vec, y)
        unsupervised_metrics_viz(M, [m.__name__ for m in metrics], types)
