#===============================Final Kaggle submission====================================================


import pandas as pd
import os
import random
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from sklearn.metrics import f1_score,classification_report,confusion_matrix,roc_auc_score
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

file_path=os.path.join(Path(__file__).resolve().parent.parent,'data/raw')
train=pd.read_csv(os.path.join(file_path,'train.csv'),index_col='ID')
test=pd.read_csv(os.path.join(file_path,'test.csv'))
sample=pd.read_csv(os.path.join(file_path,'sample_submission.csv'))

numerical_col=['donor_age','age_at_hct']
target_col=['year_hct','efs_time','efs']
categorical_col=train.drop(numerical_col+target_col,axis=1).columns

# Train preprocessing===============================

encoding=OneHotEncoder()
df_cat=pd.DataFrame(encoding.fit_transform(train[categorical_col]).toarray(),columns=encoding.get_feature_names_out())

ss=StandardScaler()
df_num=pd.DataFrame(ss.fit_transform(train[numerical_col]),columns=ss.get_feature_names_out())

pca=PCA(n_components=3,random_state=42)
df_pca=pd.DataFrame(pca.fit_transform(df_cat),columns=[f'pca_{i}' for i in range(3)])

# umap_reducer=umap.UMAP(n_components=3)
# df_umap=pd.DataFrame(umap_reducer.fit_transform(df_cat),columns=[f'umap_{i}' for i in range(3)])

df_train=pd.concat([df_cat,df_pca,df_num,train[target_col]],axis=1)

# Test Preprocessing==============================
# encoding=OneHotEncoder()
df_test_cat=pd.DataFrame(encoding.transform(test[categorical_col]).toarray(),columns=encoding.get_feature_names_out())

# ss=StandardScaler()
df_test_num=pd.DataFrame(ss.transform(test[numerical_col]),columns=ss.get_feature_names_out())

# pca=PCA(n_components=2,random_state=42)
df_test_pca=pd.DataFrame(pca.transform(df_test_cat),columns=[f'pca_{i}' for i in range(3)])

# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# df_test_tsne=pd.DataFrame(tsne.transform(df_test_cat),columns=[f'tsne_{i}' for i in range(2)])

# umap_reducer=umap.UMAP(n_components=2)
# df_test_umap=pd.DataFrame(umap_reducer.transform(df_test_cat),columns=[f'umap_{i}' for i in range(2)])
df_test=pd.concat([df_test_cat,df_test_pca,df_test_num,test['year_hct']],axis=1)


# Drop Features==================================

rm_features=[]
for i in df_train.columns:  
    try:
        score=df_train[i].value_counts()[0]/df_train[i].value_counts()[1]
        if score<100:
            continue
        else:
            rm_features.append(i)
    except:
        continue


sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)

df=df_train.drop(rm_features,axis=1).copy()
X=df.drop(['efs_time','efs'],axis=1)

print(X.shape)

## create new features========================

X['donor_age-age_at_hct']=df['donor_age']-df['age_at_hct']
X['donor_age+age_at_hct']=df['donor_age']+df['age_at_hct']
X['donor_age/age_at_hct']=df['donor_age']/df['age_at_hct']
X['donor_age*age_at_hct']=df['donor_age']*df['age_at_hct']

X.columns=[f'feature_{i}' for i in range(X.shape[1])]
X=X.fillna(-999)
y=df.efs
groups=train.race_group


df_test=df_test.drop(rm_features,axis=1)
df_test['donor_age-age_at_hct']=df_test['donor_age']-df_test['age_at_hct']
df_test['donor_age+age_at_hct']=df_test['donor_age']+df_test['age_at_hct']
df_test['donor_age/age_at_hct']=df_test['donor_age']/df_test['age_at_hct']
df_test['donor_age*age_at_hct']=df_test['donor_age']*df_test['age_at_hct']

df_test.columns=[f'feature_{i}' for i in range(X.shape[1])]
df_test=df_test.fillna(-999)

print(df_test.shape,X.shape)


lgb_params={
    "boosting_type": "gbdt",
    'random_state': 42,  "max_depth": 9,"learning_rate": 0.1,
    "n_estimators": 768,"colsample_bytree": 0.6,"colsample_bynode": 0.6,
    "verbose": -1,"reg_alpha": 0.2,
    "reg_lambda": 5,"extra_trees":True,'num_leaves':64,"max_bin":255,
    'importance_type': 'gain',
    }

cat_params={
    'random_state':42,
    'bagging_temperature': 0.50,'iterations': 650,
    'learning_rate': 0.1,'max_depth': 8,
    'l2_leaf_reg': 1.25,'min_data_in_leaf': 24,
    'random_strength' : 0.25, 'verbose': 0,
            }
xgb_params={
    'random_state': 42, 'n_estimators': 256, 
    'learning_rate': 0.1, 'max_depth': 6,
    'reg_alpha': 0.08, 'reg_lambda': 0.8, 
    'subsample': 0.95, 'colsample_bytree': 0.6, 
    'min_child_weight': 3,
    'enable_categorical':True
            }

fold_pred=[]
for fold,(train_idx,val_idx) in enumerate(sgkf.split(X=X,y=y,groups=groups)):
    print(f"Fold {fold+1}:")
    X_train,X_valid,y_train,y_valid=X.loc[train_idx],X.loc[val_idx],y.loc[train_idx],y.loc[val_idx]
    df_temp=df_train.loc[y_train.index,['efs','efs_time']].copy()
    X_train=X_train.loc[(df_temp.loc[((df_temp.efs==1)&(df_temp.efs_time<7.2))|((df_temp.efs==0)&(df_temp.efs_time<200)),'efs'].index),:]
    y_train=y_train.loc[(df_temp.loc[((df_temp.efs==1)&(df_temp.efs_time<7.2))|((df_temp.efs==0)&(df_temp.efs_time<200)),'efs'].index)]


    models = {
    # "Voting_Classifier_2": Pipeline(steps=[
    #     ('voting', VotingClassifier([
    #         ('cat',CatBoostClassifier(**cat_params)),
    #         ('xgb', XGBClassifier(**xgb_params)),
    #         ('rf', RandomForestClassifier(random_state=42)),
    #         ('lgbm', LGBMClassifier(**lgb_params))
    #     ], voting='soft'))
    # ]),
    # "Voting_Classifier_3": Pipeline(steps=[
    #     ('voting', VotingClassifier([
    #         ('cat',CatBoostClassifier(random_state=42,verbose=False)),
    #         ('xgb', XGBClassifier(**xgb_params)),
    #         # ('rf', RandomForestClassifier(random_state=42)),
    #         ('lgbm', LGBMClassifier(**lgb_params))
    #     ], voting='soft'))
    # ]),
    "Voting_Classifier_4": Pipeline(steps=[
        ('voting', VotingClassifier([
            ('cat',CatBoostClassifier(**cat_params)),
            # ('xgb', XGBClassifier(random_state=42)),
            # ('rf', RandomForestClassifier(random_state=42)),
            ('lgbm', LGBMClassifier(**lgb_params))
        ], voting='soft'))
    ])
    }

    results = []

    model_pred=[]

    for model_name, model in models.items():
        print(f"Training {model_name}...")
 
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_valid)
        pred_proba = model.predict_proba(X_valid)[:, 1]

        final = train.loc[y_valid.index, ['race_group', 'efs_time', 'efs']].copy()
        final['pred'] = pred_proba
        final = final.reset_index(drop=True)

        for race in final['race_group'].unique():
            sub_df = final[final['race_group'] == race]

        race_score_rounded = np.round(score, 3).tolist()

        model_result = {
            "name": model_name,
            "metrics": {
                "f1_score": round(f1_score(y_valid, y_pred), 3),
                "roc_auc_score": round(roc_auc_score(y_valid, y_pred), 3),
            },
            
        }

        results.append(model_result)
        model_pred.append(model.predict_proba(df_test)[:,1])
    print(results)
    fold_pred.append(model_pred)



final_pred=np.array(fold_pred)
print(final_pred)
print(final_pred.shape)



## final sumbmission====================================

sub=test[['ID']].copy()
sub['prediction']=final_pred.mean(axis=(0,1))
# sub.to_csv('submission.csv',index=False)