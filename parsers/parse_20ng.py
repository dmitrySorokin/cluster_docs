from sklearn.datasets import fetch_20newsgroups
import sqlite3
from tqdm import tqdm

from utils import cleanup, normalize


def write_text(cur, file_id, file_path, label, text):
    text = cleanup(text)
    text = normalize(text)
    cur.execute(
            'INSERT INTO Files ('
            'file_id, file_path, label_ids, text) VALUES '
            '({}, "{}", "{}", "{}")'.format(
                file_id, 
                file_path, 
                label, 
                text
    ))


if __name__  == '__main__':
    newsgroups = fetch_20newsgroups(subset='all') #remove=('headers', 'footers', 'quotes')

    conn = sqlite3.connect('20ng.sqlite')
    cursor = conn.cursor()

    cursor.execute(
        'CREATE TABLE Files('
        'file_id INTEGER NOT NULL PRIMARY KEY, '
        'file_path TEXT NOT NULL, '
        'label_ids TEXT NOT NULL, '
        'text TEXT NOT NULL)')
    cursor.execute(
        'CREATE TABLE Labels('
        'label_id INTEGER NOT NULL PRIMARY KEY, '
        'label_desc TEXT NOT NULL)'
    )
    conn.commit()

    total = len(newsgroups.data)
    print(f'total docs: {total}')

    #https://stackoverflow.com/questions/61240805/matching-target-to-target-names-in-fetch-20newsgroups
    label2desc = {}
    for label_id, labeldesc in enumerate(newsgroups.target_names):
        label2desc[label_id] = labeldesc
    print('total labels:', len(label2desc))

    for label_id in label2desc:
        cursor.execute(
            'INSERT INTO Labels ('
            'label_id, label_desc) VALUES '
            '({}, "{}")'.format(label_id, label2desc[label_id])
        )
    conn.commit()

    for i, (text, fname, label_id) in tqdm(enumerate(zip(newsgroups.data, newsgroups.filenames, newsgroups.target)), total=total):
        write_text(cursor, i, fname, label_id, text)
        conn.commit()
