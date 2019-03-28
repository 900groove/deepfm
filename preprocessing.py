import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


"""
学習データを前処理しする
与えられた学習データは0~13カラムが連続値、それ以降はカテゴリデータ
連続値は任意の区間でピン分割しカテゴリ化
すべてのカテゴリのユニーク数をfeature_sizeとして保存
"""

df = pd.read_csv('./data/raw/train.txt', delimiter='\t', header=None)

indexer = {}
feature_size = []

# カテゴリ値をラベルエンコーディングする
# 連続値は最大区間を100に分けた後、カテゴリ値として上記の処理を行う

# 学習データの1カラム目はtarget
for col in df.columns[1:]:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        indexer[col] = le
        feature_size.append(len(le.classes_))
    else:
        df[col] = pd.cut(df[col], 100, labels=False)
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        indexer[col] = le
        feature_size.append(len(le.classes_))

df.to_csv('./data/processed/train.csv')

f = open('./data/processed/feature_size.pkl', 'wb')
pickle.dump(feature_size, f)