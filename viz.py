import numpy as np
import altair as alt
import pandas as pd
import os
import argparse
from sklearn.metrics.cluster import *
import sqlite3


from vectorize import cluster


def contingency_matrix(y, y_pred, df=pd.DataFrame(), X=None, X_pred=None,
                       y_labels=None, y_pred_labels=None,
                       tooltip_cols=None,
                       table_cols=None, table_widths=None,
                       width=600, height=600, href=None,
                       cmap='tableau20', filename='cm.html',
                       sort=True, sort_type='rc',
                       inter_type='mat_leg'):
    tooltip_cols = [] if tooltip_cols is None else tooltip_cols
    table_cols = [] if table_cols is None else table_cols
    table_widths = [0 for _ in table_cols] if table_widths is None else table_widths
    tooltip_cols.extend(['label_id', 'label_id_pred'])
    X = X_pred if X is None else X
    df['x'], df['y'] = X[:, 0], X[:, 1]
    df['x2'], df['y2'] = X_pred[:, 0], X_pred[:, 1]
    df['label_id'], df['label_id_pred'] = y, y_pred
    if y_labels is not None:
        df['label'] = y_labels
        tooltip_cols.append('label')
        label_col = 'label'
        y_i2l = dict(zip(y, y_labels))
    else:
        label_col = 'label_id'
    if y_pred_labels is not None:
        df['label_pred'] = y_pred_labels
        tooltip_cols.append('label_pred')
        label_pred_col = 'label_pred'
        yp_i2l = dict(zip(y_pred, y_pred_labels))
    else:
        label_pred_col = 'label_id_pred'
    if sort:  # sort y_pred
        labels_ids = sorted(list(set(y)))
        labels_pred_ids = sorted(list(set(y_pred)))
        # print(labels_ids)
        # print(labels_pred_ids)
        l2i = dict(zip(labels_ids, range(len(labels_ids))))
        lp2i = dict(zip(labels_pred_ids, range(len(labels_pred_ids))))
        cm = np.zeros((len(labels_ids), len(labels_pred_ids)))
        for label_id, label_id_pred in zip(y, y_pred):
            cm[l2i[label_id], lp2i[label_id_pred]] += 1
        print(cm)
        # y_order = np.argsort(-np.amax(cm, axis=1))
        # y_pred_order = np.argsort(-np.amax(cm, axis=0))
        if sort_type == 'rc':
            new_rows = np.lexsort((np.argmax(cm, axis=1),
                                   -np.amax(cm, axis=1)))
            new_cm_rc = cm[new_rows, :]
            new_cols = np.lexsort((np.argmax(new_cm_rc, axis=0),
                                   -np.amax(new_cm_rc, axis=0)))
            new_cm_rc = new_cm_rc[:, new_cols]
            print(new_cm_rc)
        elif sort_type == 'cr':
            new_cols = np.lexsort((np.argmax(cm, axis=0),
                                   -np.amax(cm, axis=0)))
            new_cm_cr = cm[:, new_cols]
            new_rows = np.lexsort((np.argmax(new_cm_cr, axis=1),
                                   -np.amax(new_cm_cr, axis=1)))
            new_cm_cr = new_cm_cr[new_rows, :]
            print(new_cm_cr)
        else:
            raise ValueError(sort_type)
        y_order = new_rows
        y_pred_order = new_cols
        y_order = [labels_ids[i] for i in y_order]
        y_pred_order = [labels_pred_ids[i] for i in y_pred_order]
        if y_labels is not None:
            y_order = [y_i2l[x] for x in y_order]
        if y_pred_labels is not None:
            y_pred_order = [yp_i2l[x] for x in y_pred_order]
    else:
        y_order = 'ascending'
        y_pred_order = 'ascending'
    # print(df[label_col])
    if inter_type == 'mat_leg':
        selection = alt.selection_multi(
            fields=[label_col, label_pred_col],
            empty='none',
            init=[{label_col: v, label_pred_col: v2}
                  for v, v2 in zip(list(df[label_col]),
                                   list(df[label_pred_col]))]
            # toggle='event.altKey'
            # toggle='event.altKey && event.shiftKey'
            )
        sel_y = alt.selection_multi(fields=[label_col],
                                    bind='legend',
                                    empty='none',
                                    toggle='event.altKey'
                                    )
        sel_y_pred = alt.selection_multi(fields=[label_pred_col],
                                         bind='legend',
                                         empty='none',
                                         toggle='event.altKey'
                                         )
        base = alt.Chart(df).transform_aggregate(
            cm='count()',
            groupby=[label_col, label_pred_col]
        ).encode(
            alt.X(label_pred_col + ':N', scale=alt.Scale(paddingInner=0),
                  sort=y_pred_order),
            alt.Y(label_col + ':N', scale=alt.Scale(paddingInner=0),
                  sort=y_order),
        ).properties(
            width=width,
            height=height
        )
        legend = base.mark_rect().encode(
            color=alt.condition(selection | sel_y | sel_y_pred,
                                alt.Color('cm:Q', legend=None),
                                alt.value('lightgray')),
        ).add_selection(
            selection
        )
        text = base.mark_text(baseline='middle').encode(
            text='cm:Q',
            color=alt.value('black'),
            # color=alt.condition(
            #     alt.datum.num_cars > 100,
            #     alt.value('black'),
            #     alt.value('white')
            # )
        )
        points = alt.Chart(df).mark_point(filled=True).encode(
            alt.X('x:Q', axis=None),
            alt.Y('y:Q', axis=None),
            color=alt.condition(selection | sel_y | sel_y_pred,
                                alt.Color(label_col + ':N', sort=y_order,
                                          scale=alt.Scale(scheme=cmap)),
                                alt.value('lightgray')),
            opacity=alt.condition(selection | sel_y | sel_y_pred,
                                  alt.value(1.0), alt.value(0.3)),
            tooltip=tooltip_cols
        ).add_selection(
            sel_y
        ).properties(
            width=width / 2,
            height=height / 2
        ).interactive()
        if href is not None:
            points = points.encode(href=href + ':N')
        points_pred = alt.Chart(df).mark_point(filled=True).encode(
            alt.X('x2:Q', axis=None),
            alt.Y('y2:Q', axis=None),
            color=alt.condition(selection | sel_y | sel_y_pred,
                                alt.Color(label_pred_col + ':N',
                                          sort=y_pred_order,
                                          scale=alt.Scale(scheme=cmap)),
                                alt.value('lightgray')),
            opacity=alt.condition(selection | sel_y | sel_y_pred,
                                  alt.value(1.0), alt.value(0.3)),
            tooltip=tooltip_cols
        ).properties(
            width=width / 2,
            height=height / 2
        ).add_selection(
            sel_y_pred
        ).interactive()
        if href is not None:
            points_pred = points_pred.encode(href=href + ':N')
        tables = []
        for c, w in zip(table_cols, table_widths):
            tables.append(alt.Chart(df).mark_text().encode(
                text=c + ':N',
                y=alt.Y('row_number:O', axis=None)
            ).transform_window(
                row_number='row_number()'
            ).transform_filter(
                selection | sel_y | sel_y_pred
            ).transform_window(
                rank='rank(row_number)'
            ).properties(title=c, width=w))
    else:
        raise ValueError(inter_type)
    table = alt.hconcat(*tables)
    cmat = ((legend + text | (points & points_pred).resolve_scale(
        color='independent')) & table)
    cmat.save(filename)
    return cmat


def metrics(M, metrics_labels, types, filename='metrics.html'):
    rows = [(metrics_labels[i], types[j], types[k], m)
            for (i, j, k), m in np.ndenumerate(M)]
    columns = ['m', 't1', 't2', 'v']
    df = pd.DataFrame.from_records(rows, columns=columns)
    rating_radio = alt.binding_radio(options=metrics_labels)
    rating_select = alt.selection_single(fields=['m'],
                                         bind=rating_radio,
                                         name="Metric",
                                         empty='none',
                                         init={'m': metrics_labels[0]})
    heatmap = alt.Chart(df).encode(
        alt.X('t2:N'),
        alt.Y('t1:N'),
    ).mark_rect().encode(
        color='v:Q'
    )
    text = alt.Chart(df).encode(
        alt.X('t2:N'),
        alt.Y('t1:N'),
    ).mark_text(baseline='middle').encode(
        text=alt.Text('v:Q', format='.3f'),
        color=alt.value('black'),
        # color=alt.condition(
        #     alt.datum.v < 0.8,
        #     alt.value('black'),
        #     alt.value('white')
        # )
    )
    p = (heatmap + text).add_selection(
        rating_select
    ).transform_filter(
        rating_select
    ).properties(width=400, height=400)
    p.save(filename)
    return p


def unsupervised_metrics(M, metrics_labels, types,
                         filename='metrics_unsuper.html'):
    rows = [(metrics_labels[i], types[j], m)
            for (i, j), m in np.ndenumerate(M)]
    columns = ['m', 't', 'v']
    df = pd.DataFrame.from_records(rows, columns=columns)
    heatmap = alt.Chart(df).encode(
        alt.X('m:N'),
        alt.Y('t:N'),
    ).mark_rect().encode(
        color='v:Q'
    )
    text = alt.Chart(df).encode(
        alt.X('m:N'),
        alt.Y('t:N'),
    ).mark_text(baseline='middle').encode(
        text=alt.Text('v:Q', format='.3f'),
        color=alt.value('black'),
        # color=alt.condition(
        #     alt.datum.v < 0.8,
        #     alt.value('black'),
        #     alt.value('white')
        # )
    )
    p = (heatmap + text).properties(width=400, height=400)
    p.save(filename)
    return p


def read(conn, name):
    df = pd.read_sql(f'SELECT vec FROM {name}', conn)
    vectors = []
    for v in df['vec']:
        vectors.append(list(map(float, v.split(','))))
    return np.asarray(vectors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['word2vec', 'word2vec_idf', 'doc2vec', 'lsa', 'lda', 'rdf', 'topic_net'])
    parser.add_argument('--type2', choices=['word2vec', 'word2vec_idf', 'doc2vec', 'lsa', 'lda', 'rdf', 'topic_net'], default=None)
    parser.add_argument('--cmat', action='store_true')
    parser.add_argument('--metrics', action='store_true')

    args = parser.parse_args()

    conn = sqlite3.connect('data/mouse.sqlite')
    files = pd.read_sql('SELECT * FROM Files', conn)

    vectors = read(conn, args.type)
    labels = cluster(vectors, 27)

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
            X_pred = np.array([vectors[i] for i in new_ids])
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
            vectors2 = read(conn, args.type2)
            labels2 = cluster(vectors2, n_clusters=20)
            y = labels
            y_labels = None
            y_pred = labels2
            y_pred_labels = None
            X = vectors2
            X_pred = vectors2
            df = files
            df['file_name'] = [os.path.basename(x) for x in df['file_path']]
            df['file_path'] = ['//' + x for x in df['file_path']]
            del df['text']  # due to performance issues
            tooltip_cols = ['file_id', 'file_name', 'file_path']
            table_cols = ['file_id', 'file_name', 'label_id', 'label_id_pred']
            href = 'file_path'
            cm_filename = f'cm_{args.type}_{args.type2}.html'
        contingency_matrix(y=y,
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
        metrics(M, [m.__name__ for m in metrics], types)
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
        unsupervised_metrics(M, [m.__name__ for m in metrics], types)
