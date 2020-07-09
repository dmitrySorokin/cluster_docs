import requests
from bs4 import BeautifulSoup 
import wget
import csv
from collections import defaultdict
import zipfile36 as zipfile
import os
import sqlite3
from tqdm import tqdm

from utils import cleanup, normalize


TOPIS_URL = 'https://dmice.ohsu.edu/trec-gen/data/2006/scripts/final.goldstd.tsv.txt'
DOCS_URL = 'https://dmice.ohsu.edu/trec-gen/data/2006/documents/'


def download_zips(path='trec_gen/zips'):
    r = requests.get(DOCS_URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    for a in filter(lambda tag: tag['href'].endswith('.zip'), soup.find_all('a', href=True)):
        while True:
            try:
                wget.download(DOCS_URL + a['href'], out=path)
            except Exception as e:
                print(e)
            else:
                break


def unzip(path_from='trec_gen/zips', path_to='trec_gen/files'):
    for fname in os.listdir(path_from):
        with zipfile.ZipFile(os.path.join(path_from, fname), 'r') as zip_ref:
            zip_ref.extractall(path_to)


def write_text(cur, file_id, file_path, label, text, start, length):
    text = cleanup(text)
    text = normalize(text)
    cur.execute(
            'INSERT INTO Files ('
            'file_id, file_path, label_ids, text, start, length) VALUES '
            '({}, "{}", "{}", "{}", {}, {})'.format(
                file_id, 
                file_path, 
                label, 
                text,
                start, 
                length
            ))


if __name__ == '__main__':
    #download_zips()
    #unzip()

    conn = sqlite3.connect('trecgen2.sqlite')
    cursor = conn.cursor()

    cursor.execute(
        'CREATE TABLE Files('
        'file_id INTEGER NOT NULL PRIMARY KEY, '
        'file_path TEXT NOT NULL, '
        'label_ids TEXT NOT NULL, '
        'text TEXT NOT NULL, '
        'start INTEGER NOT NULL, '
        'length INTEGER NOT NULL'
        ')')
    cursor.execute(
        'CREATE TABLE Labels('
        'label_id INTEGER NOT NULL PRIMARY KEY, '
        'label_desc TEXT NOT NULL)'
    )
    conn.commit()

    label2desc = {}
    with open('trec_gen/2006topics.txt', 'r', encoding=' Windows-1252') as top:
        for line in top.read().split('\n')[:-1]:
            label2desc[int(line[1:4])] = line[5:]
    print('total labels:', len(label2desc))


    for label_id in label2desc:
        cursor.execute(
            'INSERT INTO Labels ('
            'label_id, label_desc) VALUES '
            '({}, "{}")'.format(label_id, label2desc[label_id])
        )
    conn.commit()


    id2path = {}
    root = 'trec_gen/files'
    for folder in os.listdir(root):
        for file in os.listdir(os.path.join(root, folder)):
            id2path[file[:-5]] = os.path.join(root, folder, file)
    print('total files:', len(id2path))


    texts = defaultdict(list)
    with open('trec_gen/final.goldstd.tsv.txt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for i, (label, file_id, start, length, tags) in tqdm(enumerate(reader)):
            start = int(start)
            length = int(length)
            label = int(label)
            if file_id not in id2path:
                print('missing', file_id)
            else:
                with open(id2path[file_id], 'rb') as doc:
                    #print(id2path[file_id], start, length, tags)
                    #print(f'Q: "{label2desc[label]}"')
                    text = doc.read()
                    text = text[start: start + length]
                    text = text.decode('utf8')
                    text = BeautifulSoup(text, 'html.parser').text
                    #print(f'A: "{text}"')
                    write_text(cursor, i, id2path[file_id], label, text, start, length)
                    conn.commit()

