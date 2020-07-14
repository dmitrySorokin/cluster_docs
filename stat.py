import sqlite3
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mouse', 'trecgen', '20ng'])
    args = parser.parse_args()

    conn = sqlite3.connect(f'data/{args.dataset}.sqlite')
    cursor = conn.cursor()

    labels = {}
    desc = {}
    for (l, label_desc) in cursor.execute('SELECT label_id, label_desc from Labels'):
        labels[l] = 0
        desc[l] = label_desc


    for (file_id, label_ids) in cursor.execute('SELECT file_id, label_ids FROM Files'):
        file_labels = map(int, label_ids.split(','))
        for l in file_labels:
            labels[l] += 1


    total = 0
    for l in labels:
        total += labels[l]
        print('{}: {}'.format(desc[l], labels[l]))
    print('-'* 10)
    print('total = {}'.format(total))
