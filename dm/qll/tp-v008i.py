import pandas as pd
import numpy as np

import sklearn as sk
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import imblearn as im
from imblearn import under_sampling
from imblearn import over_sampling
from collections import Counter

def accuracy_precision_recall(X, y):
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.5, shuffle=True, stratify=y, random_state=42)

    clf = sk.ensemble.RandomForestClassifier(max_features= 'log2',min_samples_split = 20, min_samples_leaf = 10, n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("accuracy:", sk.metrics.accuracy_score(y_test, y_pred))
    print("precision:", sk.metrics.precision_score(y_test, y_pred))
    print("recall:", sk.metrics.recall_score(y_test, y_pred))
    print("f1:",  sk.metrics.f1_score(y_test, y_pred))

def crear_columna_prop_ponderado(df_train, df_test, campo_group_by, campo_id, campo_label):
    df_tmp = pd.DataFrame()
    df_tmp_global = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp_global['cant_total_global'] = df_train.groupby([campo_id],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp = pd.merge(df_tmp, df_tmp_global, how='left', left_index=True, right_index=True)

    df_tmp['proporcion'] = ((df_tmp['cant_likes']-df_tmp['cant_dislikes'])*df_tmp['cant_total'])/ (df_tmp['cant_total_global'])

    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)
    
    columnas = list(df_train.groupby(campo_group_by,observed=False).groups.keys())
    
    col_list = ['cant_likes','cant_dislikes','cant_total','cant_total_global']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()
    df_pivot = df_tmp.pivot(index=campo_id, columns=campo_group_by, values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)
    
    
    columna_res = campo_group_by + '_' + 'prop_pond'
    columnas_campo_group_by = []
    for i in range(len(columnas)):
    	columnas_campo_group_by.append(campo_group_by + '_' + columnas[i])

    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')
    df_train_dummies = pd.get_dummies(df_train[campo_group_by], prefix=campo_group_by)
    df_train = pd.concat([df_train, df_train_dummies], axis=1)
    for i in range(len(columnas)):
    	df_train[columnas[i]] = df_train[[columnas[i],columnas_campo_group_by[i]]].prod(axis=1)
    df_train[columna_res] = df_train[columnas].sum(axis=1)
    df_train.drop(columns=columnas_campo_group_by, inplace=True)
    df_train.drop(columns=columnas, inplace=True)
    
    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')
    df_test_dummies = pd.get_dummies(df_test[campo_group_by], prefix=campo_group_by)
    df_test = pd.concat([df_test, df_test_dummies], axis=1)
    for i in range(len(columnas)):
    	df_test[columnas[i]] = df_test[[columnas[i],columnas_campo_group_by[i]]].prod(axis=1)
    df_test[columna_res] = df_test[columnas].sum(axis=1)
    df_test.drop(columns=columnas_campo_group_by, inplace=True)
    df_test.drop(columns=columnas, inplace=True) 
      
    return df_train, df_test

def crear_columna_frec(df_train,df_test, lista_campos_group_by, campo_label, valor_label):
    colname = '_'.join(lista_campos_group_by)
    colname = '_'.join([colname, str(valor_label)])
    df_tmp = df_train[df_train[campo_label] == valor_label].groupby(lista_campos_group_by).size().reset_index(name=colname)
    
    df_train = pd.merge(df_train, df_tmp, on=lista_campos_group_by, how='left')
    df_train[colname] = df_train[colname].fillna(df_train[colname].mean())
    df_train[colname] = df_train[colname].astype(int)
    df_test = pd.merge(df_test, df_tmp, on=lista_campos_group_by, how='left')
    df_test[colname] = df_test[colname].fillna(df_train[colname].mean())
    df_test[colname] = df_test[colname].astype(int)
    
    return df_train, df_test

def crear_columna_frec_rel(df_train,df_test, lista_campos_group_by, campo_label, valor_label):
    colname_pre = '_'.join(lista_campos_group_by)
    colname = '_'.join([colname_pre, str(valor_label)])
    colname_total = '_'.join([colname,'total'])
    colname_frel = '_'.join([colname,'frel'])
    
    df_tmp = df_train[df_train[campo_label] == valor_label].groupby(lista_campos_group_by, observed=False).size().reset_index(name=colname)
    df_tmp_total = df_train.groupby(lista_campos_group_by, observed=False).size().reset_index(name=colname_total)
    
    
    df_tmp_full = pd.merge(df_tmp_total, df_tmp, on=lista_campos_group_by, how='left')
    df_tmp_full[colname] = df_tmp_full[colname].fillna(value=0)
    df_tmp_full[colname_frel] = df_tmp_full[colname] / df_tmp_full[colname_total]
    
    col_list = lista_campos_group_by + [colname_frel]
    df_tmp_full = df_tmp_full[col_list]

    df_train = pd.merge(df_train, df_tmp_full, on=lista_campos_group_by, how='left')
    df_test = pd.merge(df_test, df_tmp_full, on=lista_campos_group_by, how='left')
    
    df_train[colname_frel] = df_train[colname_frel].fillna(df_train[colname_frel].mean())
    df_test[colname_frel] = df_test[colname_frel].fillna(df_test[colname_frel].mean())
    
    return df_train, df_test


def crear_columna_frec_rel_ponderado(df_train,df_test, lista_campos_group_by, campo_label, valor_label):
    colname_pre = '_'.join(lista_campos_group_by)
    colname = '_'.join([colname_pre, str(valor_label)])
    colname_total_global = '_'.join([colname,'total_global'])
    colname_total = '_'.join([colname,'total'])
    colname_frel = '_'.join([colname,'frel_pond'])
    
    df_tmp = df_train[df_train[campo_label] == valor_label].groupby(lista_campos_group_by, observed=False).size().reset_index(name=colname)
    df_tmp_total = df_train.groupby(lista_campos_group_by, observed=False).size().reset_index(name=colname_total)
    df_tmp_total_global = df_train.groupby(lista_campos_group_by[0], observed=False).size().reset_index(name=colname_total_global)
  
#    df_tmp['proporcion'] = ((df_tmp['cant_likes']-df_tmp['cant_dislikes'])*df_tmp['cant_total'])/ (df_tmp['cant_total_global'])     
    
    df_tmp_full = pd.merge(df_tmp_total, df_tmp, on=lista_campos_group_by, how='left')
    df_tmp_full = pd.merge(df_tmp_full, df_tmp_total_global, on=lista_campos_group_by[0], how='left')
    df_tmp_full[colname] = df_tmp_full[colname].fillna(value=0)
    df_tmp_full[colname_frel] = (df_tmp_full[colname]- (df_tmp_full[colname_total] - df_tmp_full[colname]))*df_tmp_full[colname_total]/df_tmp_full[colname_total_global] 

    
    col_list = lista_campos_group_by + [colname_frel]
    df_tmp_full = df_tmp_full[col_list]

    df_train = pd.merge(df_train, df_tmp_full, on=lista_campos_group_by, how='left')
    df_test = pd.merge(df_test, df_tmp_full, on=lista_campos_group_by, how='left')
    
    df_train[colname_frel] = df_train[colname_frel].fillna(df_train[colname_frel].mean())
    df_test[colname_frel] = df_test[colname_frel].fillna(df_test[colname_frel].mean())
    
    return df_train, df_test


def crear_columnas_prop(df_train, df_test, campo_group_by, campo_id, campo_label):
    df_tmp = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp['proporcion'] = df_tmp['cant_likes']/ (df_tmp['cant_total'])
    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)
    
    col_list = ['cant_likes','cant_dislikes']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()
    df_pivot = df_tmp.pivot(index=campo_id, columns=campo_group_by, values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)
    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')

    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')
    
    return df_train, df_test

def crear_n_columnas_prop(df_train, df_test, campo_group_by, campo_id, campo_label, n):
    df_tmp = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp['proporcion'] = df_tmp['cant_likes']/ (df_tmp['cant_total'])
    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)
    
    colname = '_'.join([campo_group_by, 'otros'])
    
    top_n = df_train.groupby(campo_group_by).size().nlargest(n).index
    
    df_tmp = df_tmp.reset_index()
    df_tmp['grupo'] = df_tmp[campo_group_by].apply(lambda x: x if x in top_n else colname)

    col_list = ['cant_likes','cant_dislikes']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()

    df_tmp = df_tmp[~(df_tmp['grupo'] == colname)]
 
    
    df_pivot = df_tmp.pivot(index=campo_id, columns='grupo', values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)

    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')
    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')
    
    for columna in list(top_n):
        df_train[columna] = df_train[columna].fillna(0)
        df_test[columna] = df_test[columna].fillna(0)
    
    return df_train, df_test

def crear_columnas_prop_pca(df_train, df_test, campo_group_by, campo_id, campo_label, cant_componentes):
    df_tmp = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp['proporcion'] = df_tmp['cant_likes']/ (df_tmp['cant_total'])
    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)

    columnas = list(df_train.groupby(campo_group_by).groups.keys())
    
    col_list = ['cant_likes','cant_dislikes']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()
    df_pivot = df_tmp.pivot(index=campo_id, columns=campo_group_by, values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)
    df_train_backup = df_train.copy()
    df_test_backup = df_test.copy()
    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')
    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')

    df_train_filtrado = df_train[columnas].fillna(0)
    df_test_filtrado = df_test[columnas].fillna(0)
    pca = PCA(n_components = cant_componentes)
    pca.fit(df_train_filtrado)
    train_scores = pca.transform(df_train_filtrado)
    test_scores = pca.transform(df_test_filtrado)
    pca_columns = [f'{campo_group_by}_PC{i+1}' for i in range(train_scores.shape[1])]
    df_train_scores = pd.DataFrame(data=train_scores, columns=pca_columns)
    df_test_scores = pd.DataFrame(data=test_scores, columns=pca_columns)
    df_train_res = pd.concat([df_train_backup, df_train_scores], axis=1)
    df_test_res = pd.concat([df_test_backup, df_test_scores], axis=1)
    return df_train_res, df_test_res

def crear_columnas_prop_pca_acotado(df_train, df_test, campo_group_by, campo_id, campo_label, cant_componentes, n):
    df_tmp = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp['proporcion'] = df_tmp['cant_likes']/ (df_tmp['cant_total'])
    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)
    
    colname = '_'.join([campo_group_by, 'otros'])
    
    top_n = df_train.groupby(campo_group_by).size().nlargest(n).index
    columnas = list(top_n)
    df_tmp = df_tmp.reset_index()
    df_tmp['grupo'] = df_tmp[campo_group_by].apply(lambda x: x if x in top_n else colname)

    col_list = ['cant_likes','cant_dislikes']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()

    df_tmp = df_tmp[~(df_tmp['grupo'] == colname)]
    df_tmp = df_tmp.reset_index()
    df_pivot = df_tmp.pivot(index=campo_id, columns=campo_group_by, values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)
    df_train_backup = df_train.copy()
    df_test_backup = df_test.copy()
    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')
    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')
    df_train_filtrado = df_train[columnas].fillna(0)
    df_test_filtrado = df_test[columnas].fillna(0)
    pca = PCA(n_components = cant_componentes)
    pca.fit(df_train_filtrado)
    train_scores = pca.transform(df_train_filtrado)
    test_scores = pca.transform(df_test_filtrado)
    pca_columns = [f'{campo_group_by}_PC{i+1}' for i in range(train_scores.shape[1])]
    df_train_scores = pd.DataFrame(data=train_scores, columns=pca_columns)
    df_test_scores = pd.DataFrame(data=test_scores, columns=pca_columns)
    df_train_res = pd.concat([df_train_backup, df_train_scores], axis=1)
    df_test_res = pd.concat([df_test_backup, df_test_scores], axis=1)
    return df_train_res, df_test_res




def crear_columnas_prop_ponderado(df_train, df_test, campo_group_by, campo_id, campo_label):
    df_tmp = pd.DataFrame()
    df_tmp_global = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp_global['cant_total_global'] = df_train.groupby([campo_id],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp = pd.merge(df_tmp, df_tmp_global, how='left', left_index=True, right_index=True)

    df_tmp['proporcion'] = ((df_tmp['cant_likes']-df_tmp['cant_dislikes'])*df_tmp['cant_total'])/ (df_tmp['cant_total_global'])

    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)

    col_list = ['cant_likes','cant_dislikes','cant_total','cant_total_global']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()
    df_pivot = df_tmp.pivot(index=campo_id, columns=campo_group_by, values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)
    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')

    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')
    
    return df_train, df_test

def crear_n_columnas_prop_ponderado(df_train, df_test, campo_group_by, campo_id, campo_label, n):
    df_tmp = pd.DataFrame()
    df_tmp_global = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp_global['cant_total_global'] = df_train.groupby([campo_id],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp = pd.merge(df_tmp, df_tmp_global, how='left', left_index=True, right_index=True)

    df_tmp['proporcion'] = ((df_tmp['cant_likes']-df_tmp['cant_dislikes'])*df_tmp['cant_total'])/ (df_tmp['cant_total_global'])

    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)
    
    colname = '_'.join([campo_group_by, 'otros'])
    
    top_n = df_train.groupby(campo_group_by).size().nlargest(n).index
    
    df_tmp = df_tmp.reset_index()
    df_tmp['grupo'] = df_tmp[campo_group_by].apply(lambda x: x if x in top_n else colname)

    col_list = ['cant_likes','cant_dislikes']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()

    df_tmp = df_tmp[~(df_tmp['grupo'] == colname)]
 
    
    df_pivot = df_tmp.pivot(index=campo_id, columns='grupo', values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)

    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')
    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')
    
    for columna in list(top_n):
        df_train[columna] = df_train[columna].fillna(0)
        df_test[columna] = df_test[columna].fillna(0)
    
    return df_train, df_test

def crear_columnas_prop_pca_ponderado(df_train, df_test, campo_group_by, campo_id, campo_label, cant_componentes):
    df_tmp = pd.DataFrame()
    df_tmp_global = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp_global['cant_total_global'] = df_train.groupby([campo_id],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp = pd.merge(df_tmp, df_tmp_global, how='left', left_index=True, right_index=True)

    df_tmp['proporcion'] = ((df_tmp['cant_likes']-df_tmp['cant_dislikes'])*df_tmp['cant_total'])/ (df_tmp['cant_total_global'])

    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)

    columnas = list(df_train.groupby(campo_group_by,observed=False).groups.keys())
    
    col_list = ['cant_likes','cant_dislikes']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()
    df_pivot = df_tmp.pivot(index=campo_id, columns=campo_group_by, values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)
    df_train_backup = df_train.copy()
    df_test_backup = df_test.copy()
    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')
    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')

    df_train_filtrado = df_train[columnas].fillna(0)
    df_test_filtrado = df_test[columnas].fillna(0)
    pca = PCA(n_components = cant_componentes)
    pca.fit(df_train_filtrado)
    train_scores = pca.transform(df_train_filtrado)
    test_scores = pca.transform(df_test_filtrado)
    pca_columns = [f'{campo_group_by}_PC{i+1}' for i in range(train_scores.shape[1])]
    df_train_scores = pd.DataFrame(data=train_scores, columns=pca_columns)
    df_test_scores = pd.DataFrame(data=test_scores, columns=pca_columns)
    df_train_res = pd.concat([df_train_backup, df_train_scores], axis=1)
    df_test_res = pd.concat([df_test_backup, df_test_scores], axis=1)
    return df_train_res, df_test_res

def crear_columnas_prop_pca_acotado_ponderado(df_train, df_test, campo_group_by, campo_id, campo_label, cant_componentes, n):
    df_tmp = pd.DataFrame()
    df_tmp_global = pd.DataFrame()
    df_tmp['cant_total'] = df_train.groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp_global['cant_total_global'] = df_train.groupby([campo_id],observed=False)[campo_label].size()
    df_tmp['cant_likes'] = df_train[df_train[campo_label] == 1].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp['cant_dislikes'] = df_train[df_train[campo_label] == 0].groupby([campo_id, campo_group_by],observed=False)[campo_label].size()
    df_tmp = df_tmp.fillna(0)
    df_tmp = pd.merge(df_tmp, df_tmp_global, how='left', left_index=True, right_index=True)

    df_tmp['proporcion'] = ((df_tmp['cant_likes']-df_tmp['cant_dislikes'])*df_tmp['cant_total'])/ (df_tmp['cant_total_global'])

    df_tmp['proporcion'] = df_tmp['proporcion'].fillna(value=0)
    
    colname = '_'.join([campo_group_by, 'otros'])
    
    top_n = df_train.groupby(campo_group_by).size().nlargest(n).index
    columnas = list(top_n)
    df_tmp = df_tmp.reset_index()
    df_tmp['grupo'] = df_tmp[campo_group_by].apply(lambda x: x if x in top_n else colname)

    col_list = ['cant_likes','cant_dislikes']
    df_tmp.drop(columns=col_list, inplace=True)
    df_tmp = df_tmp.reset_index()

    df_tmp = df_tmp[~(df_tmp['grupo'] == colname)]
    df_tmp = df_tmp.reset_index()
    df_pivot = df_tmp.pivot(index=campo_id, columns=campo_group_by, values='proporcion')
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.fillna(0)
    df_train_backup = df_train.copy()
    df_test_backup = df_test.copy()
    df_train = pd.merge(df_train, df_pivot, on=campo_id, how='left')
    df_test = pd.merge(df_test, df_pivot, on=campo_id, how='left')
    df_train_filtrado = df_train[columnas].fillna(0)
    df_test_filtrado = df_test[columnas].fillna(0)
    pca = PCA(n_components = cant_componentes)
    pca.fit(df_train_filtrado)
    train_scores = pca.transform(df_train_filtrado)
    test_scores = pca.transform(df_test_filtrado)
    pca_columns = [f'{campo_group_by}_PC{i+1}' for i in range(train_scores.shape[1])]
    df_train_scores = pd.DataFrame(data=train_scores, columns=pca_columns)
    df_test_scores = pd.DataFrame(data=test_scores, columns=pca_columns)
    df_train_res = pd.concat([df_train_backup, df_train_scores], axis=1)
    df_test_res = pd.concat([df_test_backup, df_test_scores], axis=1)
    return df_train_res, df_test_res


DIR = "data"
df_train = pd.read_csv(f"{DIR}/entrenamiento.csv")
df_test = pd.read_csv(f"{DIR}/prueba.csv")
df_lectores  = pd.read_csv(f"{DIR}/lectores.csv")
df_libros = pd.read_csv(f"{DIR}/libros-nuevo.csv")



df_train_orig = df_train.copy()
df_test_orig = df_test.copy()

df_train_id_lector = set(df_train['id_lector'].unique())
df_test_id_lector = set(df_test['id_lector'].unique())
id_lector_unicos_train = df_train_id_lector - df_test_id_lector
id_lector_comunes = df_train_id_lector.intersection(df_test_id_lector)
df_train = df_train[df_train['id_lector'].isin(list(id_lector_comunes))]
#df_train = df_train[~df_train['id_lector'].isin(list(id_lector_unicos_train))]

df_libros['rating_value'] = df_libros['rating_value'].astype(float)
df_libros['rating_value'] = df_libros['rating_value'].apply(pd.to_numeric)
df_libros['rating_value'] = df_libros['rating_value'].fillna(df_libros['rating_value'].mean())

columnas_a_entero = ['best_rating', 'worst_rating', 'rating_count',
                    'cant_votos_1', 'cant_votos_2', 'cant_votos_3', 'cant_votos_4',
                    'cant_votos_5', 'cant_votos_6', 'cant_votos_7', 'cant_votos_8',
                    'cant_votos_9', 'cant_votos_10']
df_libros[columnas_a_entero] = df_libros[columnas_a_entero].apply(lambda x: pd.to_numeric(x, downcast='integer'))

df_libros.rename(columns={'genero': 'generolit'}, inplace=True)
df_libros.drop_duplicates(subset=['id_libro'], keep='first', inplace=True)
df_lectores['nacimiento'] = df_lectores['nacimiento'].replace(1910, np.nan)
df_lectores['nacimiento'] = df_lectores['nacimiento'].fillna(df_lectores['nacimiento'].median())
df_libros['generolit'] = df_libros['generolit'].fillna('generolit_desconocido')
df_libros['autor'] = df_libros['autor'].fillna('autor_desconocido')

df_libros['anio_edicion'] = pd.to_numeric(df_libros['anio_edicion'], errors='coerce')
df_libros['anio_edicion'] = df_libros['anio_edicion'].fillna(df_libros['anio_edicion'].median())

nbins=10
df_libros['anio_edicion_rango'] = pd.qcut(df_libros['anio_edicion'], q=nbins, labels=[str(x) for x in range(nbins)], precision=0, duplicates='drop')

df_libros.loc[df_libros['id_libro'] == 'el-gran-gatsby', 'titulo'] = df_libros.loc[df_libros['id_libro'] == 'el-gran-gatsby-2', 'titulo'].values[0]
df_libros.loc[df_libros['id_libro'] == 'el-gran-gatsby', 'autor'] = df_libros.loc[df_libros['id_libro'] == 'el-gran-gatsby-2', 'autor'].values[0]
df_libros.loc[df_libros['id_libro'] == 'el-gran-gatsby', 'generolit'] = df_libros.loc[df_libros['id_libro'] == 'el-gran-gatsby-2', 'generolit'].values[0]


df_train_extendido = pd.merge(df_train, df_lectores, on='id_lector', how='left').merge(df_libros, on='id_libro', how='left')
df_test_extendido = pd.merge(df_test, df_lectores, on='id_lector', how='left').merge(df_libros, on='id_libro', how='left')

df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='varios', value='Varios') 
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Histórica Y Aventuras', value='Histórica y aventuras')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='histórica y aventuras', value='Histórica y aventuras')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Biografiás, Memorias', value='Biografías, Memorias')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Clasicos de la literatura', value='Clásicos de la literatura')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Novela Negra, Intriga, Terror', value='Novela negra')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Novela negra, intriga, terror', value='Novela negra')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='No Ficción', value='No ficción')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Ensayo', value='Estudios y ensayos')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Economía financiera', value='Economía')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Biografías', value='Biografías, Memorias')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Poesía, teatro', value='Poesía, Teatro')

df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Actores', value='Varios')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Administración y dirección empresarial', value='Empresa')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Autoayuda', value='Autoayuda Y Espiritualidad')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Ciencias Políticas Y Sociales', value='Ciencias Humanas')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Cómics', value='Cómics, Novela Gráfica')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Derecho', value='Ciencias Humanas')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Didáctica y metodología', value='Ciencias Humanas')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Dietética y nutrición', value='Varios')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Fotografía', value='Varios')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Guías De Viaje', value='Varios')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Historia moderna de España', value='Historia')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Histórica', value='Narrativa histórica')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Matemáticas divulgativas', value='Ciencias')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Medicina', value='Ciencias')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Naturaleza y ciencia', value='Ciencias')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Poesía', value='Poesía, Teatro')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Policiaca. Novela negra en bolsillo', value='Novela negra')
df_train_extendido.generolit = df_train_extendido.generolit.replace(to_replace='Televisión', value='Varios')

df_train_extendido['generolit'] = df_train_extendido['generolit'].astype('category')
df_train_extendido['genero'] = df_train_extendido['genero'].astype('category')


df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Romántica, Erótica', value='Romántica, erótica')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='varios', value='Varios') 
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Histórica Y Aventuras', value='Histórica y aventuras')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='histórica y aventuras', value='Histórica y aventuras')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Biografiás, Memorias', value='Biografías, Memorias')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Clasicos de la literatura', value='Clásicos de la literatura')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Novela Negra, Intriga, Terror', value='Novela negra')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Novela negra, intriga, terror', value='Novela negra')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='No Ficción', value='No ficción')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Ensayo', value='Estudios y ensayos')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Economía financiera', value='Economía')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Biografías', value='Biografías, Memorias')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Poesía, teatro', value='Poesía, Teatro')

df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Informática', value='Ciencias')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Medicina divulgativa', value='Varios')
df_test_extendido.generolit = df_test_extendido.generolit.replace(to_replace='Política nacional', value='Varios')

df_test_extendido['generolit'] = df_test_extendido['generolit'].astype('category')
df_test_extendido['genero'] = df_test_extendido['genero'].astype('category')

df_train_extendido_dummies_genero = pd.get_dummies(df_train_extendido['genero'], prefix='genero', drop_first = True)
df_test_extendido_dummies_genero = pd.get_dummies(df_test_extendido['genero'], prefix='genero', drop_first = True)
df_train_extendido_dummies_generolit = pd.get_dummies(df_train_extendido['generolit'], prefix='generolit')
df_test_extendido_dummies_generolit = pd.get_dummies(df_test_extendido['generolit'], prefix='generolit')

df_train_vars, df_test_vars = df_train_extendido, df_test_extendido

df_train_extendido['autor_low'] = df_train_extendido['autor'].str.lower()
df_test_extendido['autor_low'] = df_test_extendido['autor'].str.lower()
df_train_extendido['titulo_low'] = df_train_extendido['titulo'].str.lower()
df_test_extendido['titulo_low'] = df_test_extendido['titulo'].str.lower()
df_train_extendido['editorial_low'] = df_train_extendido['editorial'].str.lower()
df_test_extendido['editorial_low'] = df_test_extendido['editorial'].str.lower()

#lista_campos = [['id_lector', 'generolit'],['id_lector','autor_low'],['generolit'],['titulo_low'],['autor_low'],['id_lector']]
#for campos in lista_campos:
#    df_train_vars, df_test_vars = crear_columna_frec(df_train_vars, df_test_vars,campos, 'label', 1)
#    df_train_vars, df_test_vars = crear_columna_frec(df_train_vars, df_test_vars,campos, 'label', 0)

lista_campos = [['generolit'],['titulo_low'],['autor_low']]

for campos in lista_campos:
    df_train_vars, df_test_vars = crear_columna_frec(df_train_vars, df_test_vars,campos, 'label', 1 )
    df_train_vars, df_test_vars = crear_columna_frec(df_train_vars, df_test_vars,campos, 'label', 0 )

lista_campos = [['id_lector','generolit'],['id_lector','autor_low'],['id_lector','editorial_low'],['id_lector','anio_edicion_rango']]

for campos in lista_campos:
    df_train_vars, df_test_vars = crear_columna_frec(df_train_vars, df_test_vars,campos, 'label', 1 )
    df_train_vars, df_test_vars = crear_columna_frec(df_train_vars, df_test_vars,campos, 'label', 0 )
    
df_train_generolit, df_test_generolit = crear_columnas_prop_ponderado(df_train_vars, df_test_vars, 'generolit', 'id_lector', 'label')
df_train_titulo, df_test_titulo = crear_columnas_prop_pca_acotado_ponderado(df_train_generolit, df_test_generolit, 'titulo_low', 'id_lector', 'label',2,1500)
df_train_autor, df_test_autor = crear_columnas_prop_pca_acotado_ponderado(df_train_titulo, df_test_titulo, 'autor_low', 'id_lector', 'label',1,500)
df_train_editorial, df_test_editorial = crear_columnas_prop_pca_ponderado(df_train_autor, df_test_autor, 'editorial_low', 'id_lector', 'label',1)
df_train_anio_edicion_rango, df_test_anio_edicion_rango = crear_columnas_prop_pca_ponderado(df_train_editorial, df_test_editorial, 'anio_edicion_rango', 'id_lector', 'label',1)


#df_train_titulo, df_test_titulo = crear_columnas_prop_pca_acotado(df_train_generolit, df_test_generolit, 'titulo_low', 'id_lector', 'label',20,df_train_extendido['titulo_low'].nunique())
#df_train_autor, df_test_autor = crear_columnas_prop_pca_acotado(df_train_titulo, df_test_titulo, 'autor_low', 'id_lector', 'label',20,df_train_extendido['autor_low'].nunique())
#df_train_editorial, df_test_editorial = crear_columnas_prop_pca_acotado(df_train_autor, df_test_autor, 'editorial_low', 'id_lector', 'label',10,df_train_extendido['editorial_low'].nunique())

df_train_extendido_con_dummies = pd.concat([df_train_anio_edicion_rango, df_train_extendido_dummies_genero, df_train_extendido_dummies_generolit], axis=1)
df_test_extendido_con_dummies = pd.concat([df_test_anio_edicion_rango, df_test_extendido_dummies_genero, df_test_extendido_dummies_generolit], axis=1)


df_train = df_train_extendido_con_dummies
df_test = df_test_extendido_con_dummies

df_train = df_train.select_dtypes(include=['float64', 'int64', 'int32', 'int16', 'int8', 'bool'])
#df_train = df_train.fillna(0)
#df_test = df_test.fillna(0)

#from sklearn.decomposition import PCA
#pca = PCA(n_components = 10)
#pca.fit(df_train)
#x = pca.transform(df_train)

col_list = ['id_lector','anio_edicion','best_rating', 'worst_rating', 'rating_count']

df_train.drop(columns=col_list, inplace=True)


## Datos a predecir
X = df_train[df_train.columns.drop('label')]
y = df_train['label']


# resultados = []

# for n_estimators in [50, 100, 500, 1000]:
#     for max_depth in [5, 10, 15, 30]:
#         print(f"{n_estimators=} -- {max_depth=}")

#         # Creamos el modelo
#         clf = sk.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)

#         scores_train = []
#         scores_test = []

#         # Validación cruzada, 10 folds, shuffle antes, semilla aleatoria
#         kf = sk.model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

#         for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
#             # Partimos el fold en entrenamiento y prueba...
#             X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

#             # Entrenamos el modelo en entramiento
#             clf.fit(X_train, y_train)

#             # Predecimos en train
#             y_pred = clf.predict(X_train)

#             # Medimos la performance de la predicción en entramiento
#             score_train = sk.metrics.f1_score(y_train, y_pred)
#             scores_train.append(score_train)

#             # Predecimos en test
#             y_pred = clf.predict(X_test)

#             # Medimos la performance de la predicción en prueba
#             score_test = sk.metrics.f1_score(y_test, y_pred)
#             scores_test.append(score_test)

#             print("\t", f"{fold=}, {score_train=} {score_test=}")
#             media_scores_entrenamiento = pd.Series(scores_train).mean()
#             std_scores_entrenamiento = pd.Series(scores_train).std()
#             media_scores_prueba = pd.Series(scores_test).mean()
#             std_scores_prueba=pd.Series(scores_test).std()

#         print(f"Media de scores en entrenamiento={media_scores_entrenamiento}, std={std_scores_entrenamiento}")
#         print(f"Media de scores en prueba={media_scores_prueba}, std={std_scores_prueba}")
#         print()
#         resultados.append([n_estimators, max_depth, media_scores_entrenamiento, std_scores_entrenamiento, media_scores_prueba, std_scores_prueba])


X_test = df_test[df_train.columns.drop('label')]

# Entrenamos el modelo usando todos los datos de entrenamiento
# TODO: Poner los valores de hiperparámetros que mejor dieron en el paso anterior
n_estimators = 1000
max_depth = 15
clf = sk.ensemble.RandomForestClassifier(max_features= 2,min_samples_split = 20, min_samples_leaf = 10, n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)

#enn = im.under_sampling.EditedNearestNeighbours(sampling_strategy='majority')

#X, y = enn.fit_resample(X, y)

#ros = im.over_sampling.RandomOverSampler(random_state=42, shrinkage=0.1)
#X, y = ros.fit_resample(X, y)

#clf = sk.ensemble.RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
clf.fit(X, y)

# Predecimos

df_test['label'] = clf.predict(X_test)
#df_test['label'] = clf.predict_proba(X_test)[:,1]

#proba1 = clf.predict_proba(X_test)[:,1]
#umbral = 0.4
#df_test['label'] = np.where(proba1 > umbral, 1, 0)


df_test['id'] = df_test['id_lector'].astype(str) + "--" + df_test['id_libro'].astype(str)
df_test.set_index('id', inplace=True)

# Creamos el dataframe para entregar
df_sol = df_test[["label"]]

# Tests de validación de la predicción antes de subirla
# Estos tests TIENEN que pasar sin error


assert df_sol.shape[0] == 10332, f"La cantidad de filas no es correcta. Es {df_sol.shape[0]} y debe ser 10332."
assert df_sol.shape[1] == 1, f"La cantidad de columnas no es correcta. Es {df_sol.shape[1]} y debe ser 1."
assert 'label' in df_sol.columns, "Falta la columna 'label'."
assert df_sol.index.name == 'id', "El índice debe llamarse 'id'."

version = "v008i"
df_test['label'].to_csv(f"{DIR}/solucion-{version}.csv", index=True)

imp = pd.DataFrame({
    "feature": clf.feature_names_in_,
    "importance": clf.feature_importances_
})

pd.set_option('display.max_rows', 500)
print(imp.sort_values(by="importance", ascending=False))

print((df_test['label'] == 0).sum())
#accuracy_precision_recall(X, y)



