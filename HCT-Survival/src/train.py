from pathlib import Path
import os
import pandas as pd
import random
import json

os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Adjust the number accordingly
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from sklearn.metrics import f1_score,classification_report,confusion_matrix,roc_auc_score
from lifelines.utils import concordance_index

random.seed(42)
np.random.seed(42)

print('Package loading complete....')

# file_path=Path('../data/raw')
file_path=os.path.join(Path(__file__).resolve().parent.parent,'data/processed')
train=pd.read_csv(os.path.join(file_path,'processed_train_data.csv'))
test=pd.read_csv(os.path.join(file_path,'processed_test_data.csv'))

print('data loading complete...')


rm_features=[]
for i in train.columns:  
    try:
        score=train[i].value_counts()[0]/train[i].value_counts()[1]
        if score<100:
            continue
        else:
            rm_features.append(i)
    except:
        continue


sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)

df=train.drop(rm_features,axis=1).copy()
X=df.drop(['efs_time','efs','race_group'],axis=1)

print(X.shape)

## create new features

X['donor_age-age_at_hct']=df['donor_age']-df['age_at_hct']
X['donor_age+age_at_hct']=df['donor_age']+df['age_at_hct']
X['donor_age/age_at_hct']=df['donor_age']/df['age_at_hct']
X['donor_age*age_at_hct']=df['donor_age']*df['age_at_hct']

X.columns=[f'feature_{i}' for i in range(X.shape[1])]
X=X.fillna(-999)
y=df.efs
groups=df.race_group





complete_result={
    "_comment":"efs==1&efs_time<7.2",
    "full_result":[]
}


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



for fold,(train_idx,val_idx) in enumerate(sgkf.split(X=X,y=y,groups=groups)):
    print(f"Fold {fold+1}:")
    # print("Train indices:", train_idx, "Validation indices:", val_idx)
    # print("Train classes:", y[train_idx], "Validation classes:", y[val_idx])
    # print("Groups in validation set:", np.unique(groups[val_idx]))
    X_train,X_valid,y_train,y_valid=X.loc[train_idx],X.loc[val_idx],y.loc[train_idx],y.loc[val_idx]

    df_temp=train.loc[y_train.index,['efs','efs_time']].copy()
    X_train=X_train.loc[(df_temp.loc[((df_temp.efs==1)&(df_temp.efs_time<7.2))|((df_temp.efs==0)&(df_temp.efs_time<200)),'efs'].index),:]
    y_train=y_train.loc[(df_temp.loc[((df_temp.efs==1)&(df_temp.efs_time<7.2))|((df_temp.efs==0)&(df_temp.efs_time<200)),'efs'].index)]


    models = {
    # "CatBoost": CatBoostClassifier(random_state=42, verbose=False),
    # "LGBMClassifier": LGBMClassifier(random_state=42, verbose=-1),
    # "Voting_Classifier_1": Pipeline(steps=[
    #     ('voting', VotingClassifier([
    #         ('xgb', XGBClassifier(random_state=42)),
    #         ('rf', RandomForestClassifier(random_state=42)),
    #         ('lgbm', LGBMClassifier(random_state=42, verbose=-1))
    #     ], voting='soft'))
    # ]),
    "Voting_Classifier_2": Pipeline(steps=[
        ('voting', VotingClassifier([
            ('cat',CatBoostClassifier(**cat_params)),
            ('xgb', XGBClassifier(**xgb_params)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('lgbm', LGBMClassifier(**lgb_params))
        ], voting='soft'))
    ]),
    "Voting_Classifier_3": Pipeline(steps=[
        ('voting', VotingClassifier([
            ('cat',CatBoostClassifier(random_state=42,verbose=False)),
            ('xgb', XGBClassifier(**xgb_params)),
            # ('rf', RandomForestClassifier(random_state=42)),
            ('lgbm', LGBMClassifier(**lgb_params))
        ], voting='soft'))
    ]),
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


    for model_name, model in models.items():
        print(f"Training {model_name}...")
 
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_valid)
        pred_proba = model.predict_proba(X_valid)[:, 1]

        final = train.loc[y_valid.index, ['race_group', 'efs_time', 'efs']].copy()
        final['pred'] = pred_proba
        final = final.reset_index(drop=True)

        score = []
        for race in final['race_group'].unique():
            sub_df = final[final['race_group'] == race]
            score.append(concordance_index(sub_df['efs_time'], -sub_df['pred'], sub_df['efs']))


        race_score_rounded = np.round(score, 3).tolist()

        model_result = {
            "name": model_name,
            "metrics": {
                "f1_score": round(f1_score(y_valid, y_pred), 3),
                "roc_auc_score": round(roc_auc_score(y_valid, y_pred), 3),
                "concordance_index_score": round(np.average(score), 3),
                "race_score": race_score_rounded
            },
            
        }

        results.append(model_result)
    final_results={
        "fold":fold+1,
        "result":results

    }
    complete_result['full_result'].append(final_results)

    print(results)

# print(complete_result)

with open(os.path.join(Path(__file__).resolve().parent.parent,'reports/metrics_5.json'), "w") as f:
    json.dump(complete_result, f, indent=4)

print("JSON file saved successfully.")


