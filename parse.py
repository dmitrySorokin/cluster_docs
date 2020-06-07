import sqlite3
from tqdm import tqdm
from nltk.corpus import wordnet
from multiprocessing.pool import ThreadPool
from collections import defaultdict
import os

from utils import process


_POOL = ThreadPool(processes=8)
wordnet.ensure_loaded()


KNOWLEDGE_PATH = '\\\\Trd-vm.ibrae\\создание практической методологии\\2.Расчетно-программные комплексы\\База знаний\\'
ZOTERO_MOUSE = KNOWLEDGE_PATH + 'MOUSE\\library\\zotero_profile\\zotero.sqlite'
DB_PATH = 'temp.sqlite'


if __name__ == '__main__':
    zotero_conn = sqlite3.connect(ZOTERO_MOUSE)
    zotero_cursor = zotero_conn.cursor()

    res = zotero_cursor.execute(
        'SELECT p.parentItemID AS itemId,\
          (SELECT idv.value\
           FROM itemData ii\
           JOIN itemDataValues idv ON idv.valueID = ii.valueID\
           WHERE ii.itemID = p.parentitemID\
             AND ii.fieldID = 110) AS "title",\
               ia.itemID AS attachmentItemId,\
               p.path AS filepath,\
               ci.collectionID,\
               co.collectionName\
        FROM itemAttachments p\
        JOIN items ia ON ia.itemID = p.ItemId\
        JOIN collectionItems ci ON ci.itemID=p.parentItemID\
        JOIN collections co ON ci.collectionID=co.collectionID'
        ).fetchall()


    file_labels = defaultdict(set)
    collections = {}
    for item_id, idv_value, attachment_id, file_path, collection_id, collection_name in tqdm(res):
        if file_path is not None and os.path.isfile(file_path):
            file_labels[file_path].add(collection_id)
            collections[collection_id] = collection_name


    write_conn = sqlite3.connect(DB_PATH)
    write_cursor = write_conn.cursor()

    write_cursor.execute(
        'CREATE TABLE Files('
        'file_id INTEGER NOT NULL PRIMARY KEY, '
        'file_path TEXT NOT NULL, '
        'label_ids TEXT NOT NULL, '
        'text TEXT NOT NULL)')
    write_cursor.execute(
        'CREATE TABLE Labels('
        'label_id INTEGER NOT NULL PRIMARY KEY, '
        'label_desc TEXT NOT NULL)'
    )
    write_conn.commit()

    for label_id in collections:
        write_cursor.execute(
            'INSERT INTO Labels ('
            'label_id, label_desc) VALUES '
            '({}, "{}")'.format(label_id, collections[label_id])
        )
    write_conn.commit()

    async_results = []
    for file_id, (file_path, label_ids) in enumerate(tqdm(file_labels.items())):
        future = _POOL.apply_async(process, [file_id, file_path, label_ids])
        async_results.append(future)

    for future in tqdm(async_results):
        file_id, file_path, label_ids, text = future.get()
        print(file_id, label_ids)
        write_cursor.execute(
            'INSERT INTO Files ('
            'file_id, file_path, label_ids, text) VALUES '
            '({}, "{}", "{}", "{}")'.format(
                file_id, file_path, ','.join(map(str, label_ids)), text
            ))
        write_conn.commit()
