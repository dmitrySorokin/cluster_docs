import sqlite3
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score


def read(conn, name):
    df = pd.read_sql(f'SELECT vec, label_ids FROM {name} JOIN Files ON {name}.file_id == Files.file_id', conn)
    vectors = []
    for v in df['vec']:
        vectors.append(list(map(float, v.split(','))))

    labels = []
    unique = set()
    count = defaultdict(int)
    for lbls in df['label_ids']:
        labels.append(list(map(int, lbls.split(','))))
        for l in labels[-1]:
            unique.add(l)
            count[l] += 1

    #for l in count:
    #    nm = conn.cursor().execute(f'SELECT label_desc FROM Labels WHERE label_id = {l}').fetchone()
    #    print(nm, count[l])

    ohe = OneHotEncoder()
    ohe.fit(np.asarray(list(unique)).reshape(-1, 1))
    for i, lbls in enumerate(labels):
        labels[i] = [0] * len(unique)
        for l in lbls:
            labels[i] += ohe.transform([[l]]).toarray()[0]

    return np.asarray(vectors), np.asarray(labels)


def train_test_split(vectors, labels, test_size, random_state):
    lbl2vec = defaultdict(list)
    for vec, lbls in zip(vectors, labels):
        for i, l in enumerate(lbls):
            if l:
                lbl2vec[i].append(vec)

    vectors_train = []
    labels_train = []
    vectors_test = []
    labels_test = []
    for l in lbl2vec:
        vec = lbl2vec[l]
        if len(vec) > 1:
            v_train, v_test, l_train, l_test = tts(
                vec, [l] * len(vec), test_size=test_size, random_state=random_state
            )
            vectors_train.extend(v_train)
            vectors_test.extend(v_test)
            labels_train.extend(l_train)
            labels_test.extend(l_test)

    return np.asarray(vectors_train), np.asarray(vectors_test), np.asarray(labels_train), np.asarray(labels_test)




if __name__ == '__main__':
    conn = sqlite3.connect('data/mouse.sqlite')
    vector_names = ['word2vec', 'word2vec_tfidf', 'word2vec_idf', 'doc2vec', 'lsa', 'lda', 'rdf', 'topic_net']
    for name in vector_names:
        v, true_labels = read(conn, name)
        #print(true_labels.shape)
        #print(true_labels[0])
        v_train, v_test, l_train, l_test = train_test_split(v, true_labels, test_size=0.5, random_state=0)
        #parameters = {'C': [1, 10]}
        #total_len = sum(map(len, v_train))
        #clf = OneVsRestClassifier(GridSearchCV(SVC(probability=True, class_weight='balanced'), parameters))
        clf = OneVsRestClassifier(SVC(probability=True, class_weight='balanced'))
        #clf = KNeighborsClassifier(n_neighbors=10, metric='euclidean')

        clf.fit(v_train, l_train)
        #pred = clf.predict_proba(v_test)
        #loss = log_loss(l_test, pred)
        l_pred = clf.predict(v_test)
        print('{}, {:.2f}'.format(
            name, f1_score(l_test, l_pred, average="weighted"))
        )
        #print(name, loss)
        #exit(0)

