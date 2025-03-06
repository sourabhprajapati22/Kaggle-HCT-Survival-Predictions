from pathlib import Path
import os
import pandas as pd

os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Adjust the number accordingly
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import umap
import numpy as np
import joblib

print('Package loading complete....')

# file_path=Path('../data/raw')
file_path=os.path.join(Path(__file__).resolve().parent.parent,'data/raw')
train=pd.read_csv(os.path.join(file_path,'train.csv'),index_col='ID')
test=pd.read_csv(os.path.join(file_path,'test.csv'))
sample=pd.read_csv(os.path.join(file_path,'sample_submission.csv'))
# data=pd.read_csv(os.path.join(file_path,'data_dictionary.csv'))

print('data loading complete...')

numerical_col=['donor_age','age_at_hct']
target_col=['year_hct','efs_time','efs','race_group']
categorical_col=train.drop(numerical_col+target_col,axis=1).columns


encoding=OneHotEncoder()
df_cat=pd.DataFrame(encoding.fit_transform(train[categorical_col]).toarray(),columns=encoding.get_feature_names_out())

ss=StandardScaler()
df_num=pd.DataFrame(ss.fit_transform(train[numerical_col]),columns=ss.get_feature_names_out())

pca=PCA(n_components=2,random_state=42)
df_pca=pd.DataFrame(pca.fit_transform(df_cat),columns=[f'pca_{i}' for i in range(2)])

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# df_tsne=pd.DataFrame(tsne.fit_transform(df_cat),columns=[f'tsne_{i}' for i in range(2)])


umap_reducer=umap.UMAP(n_components=2)
df_umap=pd.DataFrame(umap_reducer.fit_transform(df_cat),columns=[f'umap_{i}' for i in range(2)])

print('feature reducing complete...')

df=pd.concat([df_cat,df_umap,df_num,train[target_col]],axis=1)
# df.columns=[f'feature_{i}' for i in range(df.shape[1])]

final_path=os.path.join(Path(__file__).resolve().parent.parent,'data/processed')
df.to_csv(os.path.join(final_path,'processed_train_data.csv'),index=False)
print('train data processing complete.')

######Test data preprocessing-------------------------------------------------------------------------------------

# encoding=OneHotEncoder()
df_test_cat=pd.DataFrame(encoding.transform(test[categorical_col]).toarray(),columns=encoding.get_feature_names_out())

# ss=StandardScaler()
df_test_num=pd.DataFrame(ss.transform(test[numerical_col]),columns=ss.get_feature_names_out())

# pca=PCA(n_components=2,random_state=42)
df_test_pca=pd.DataFrame(pca.transform(df_test_cat),columns=[f'pca_{i}' for i in range(2)])

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# df_test_tsne=pd.DataFrame(tsne.transform(df_test_cat),columns=[f'tsne_{i}' for i in range(2)])

# umap_reducer=umap.UMAP(n_components=2)
df_test_umap=pd.DataFrame(umap_reducer.transform(df_test_cat),columns=[f'umap_{i}' for i in range(2)])

df_test=pd.concat([df_test_cat,df_test_umap,df_test_num,test['year_hct']],axis=1)
df_test.to_csv(os.path.join(final_path,'processed_test_data.csv'),index=False)
print('test data processing complete.')