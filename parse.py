import sqlite3
from tqdm import tqdm
from nltk.corpus import wordnet
from multiprocessing.pool import ThreadPool

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


    write_conn = sqlite3.connect(DB_PATH)
    write_cursor = write_conn.cursor()

    write_cursor.execute(
        'CREATE TABLE Files('
        'file_id INTEGER NOT NULL PRIMARY KEY, '
        'item_id INTEGER NOT NULL, '
        'collection_id INTEGER NOT NULL, '
        'collection_name TEXT NOT NULL, '
        'text TEXT NOT NULL)')
    write_conn.commit()

    async_results = []

    for file_id, (item_id, idv_value, attachment_id, file_path, collection_id, collection_name) in enumerate(tqdm(res)):
        future = _POOL.apply_async(
            process,
            [
                file_id,
                item_id,
                collection_id,
                collection_name,
                file_path
            ]
        )
        async_results.append(future)

    for future in tqdm(async_results):
        file_id, item_id, collection_id, collection_name, content = future.get()
        print(file_id, item_id, collection_id, collection_name)
        write_cursor.execute(
            'INSERT INTO Files ('
            'file_id, item_id, collection_id, collection_name, text) VALUES '
            '({}, {}, {}, "{}", "{}")'.format(file_id, item_id, collection_id, collection_name, content))
        write_conn.commit()
