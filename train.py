from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

from model.tsk_pipe import TSKPipe
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
from utils.util import *


file_name = 'Dexter'
file_path = f"data/{file_name}.csv"

original_data = pd.read_csv(file_path, header=None)

last_column = original_data.columns[-1]

original_data.fillna(original_data.mean(), inplace=True)

features = original_data.iloc[:, :-1].values
labels_init = original_data.iloc[:, -1].values

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# data preprocessing
min_max = MinMaxScaler(feature_range = (0,1))
standard = StandardScaler()

# standardize the features
features_standard = standard.fit_transform(features)
features_min_max = min_max.fit_transform(features)
labels = labels_init

num_features = features_standard.shape[1]
num_class = len(np.unique(labels)) # type: ignore

print(file_name)

rule_num = 30
alpha_threhold = 0.001
accuracy_result = {}
for normalization_method in [  #,'Z_score','Min_max'
    'Z_score','Min_max'
]:

    
    random_seed = 42
    set_seed(random_seed)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    ks_num = 0

    print(
        f"dataset: {file_name},-------------------------------------------------------------------------------------record:"
    )
    for train_index, test_index in kf.split(features_standard):
        ks_num += 1
        print(
            "-------------------------------------------------------------------------------------K-fold:",
            ks_num,
        )

        group_name = f"fold_{ks_num}" + '_217'



        X_train, X_test = (
            features_standard[train_index],
            features_standard[test_index],
        )
        X_train_minmax, X_test_minmax = (
            features_min_max[train_index],
            features_min_max[test_index],
        ) # data for HDFIS

        y_train, y_test = labels[train_index], labels[test_index]

        # all minmax
        if normalization_method == 'Min_max':
            print("all minmax")
            X_train = X_train_minmax
            X_test = X_test_minmax
        else:
        # all mean-variance
            print("all Z_score")


        # -----------------------------------------------------
        # AutoTSK
        set_seed(random_seed)
        print("AutoTSK")
        AutoTSK = TSKPipe(n_rule=rule_num, 
                            lr=0.001, 
                            weight_decay=1e-8,
                            order=1, 
                            n_class=num_class, 
                            epochs=300, 
                            patience=20, 
                            verbose=1, 
                            device='cuda', 
                            alpha_threhold = alpha_threhold
                            )
        # Search
        AutoTSK.fit_autoTSK_search(X_train.copy(), y_train.copy(), file_name=file_name, 
                                    with_LN=True)
        # AutoTSK.save_model(f'{file_name}_search2_autoTSK.pth')
        # Retrain
        AutoTSK.fit_autoTSK_retrain(X_train.copy(), y_train.copy(), 
                                    gate_for_consequent=True,
                                    file_name=file_name)


        # save model
        # AutoTSK.save_model(f'{file_name}_retrain_autoTSK.pth')
        y_pred = AutoTSK.predict(
            X_test.copy()
        )
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        add_result(accuracy_result, f"AutoTSK_with_{normalization_method}", accuracy)
        del AutoTSK
        

print(accuracy_result)
print('AutoTSK_with_Min_max:',accuracy_result['AutoTSK_with_Min_max'] , np.mean(accuracy_result['AutoTSK_with_Min_max']))
print('AutoTSK_with_Z_score:',accuracy_result['AutoTSK_with_Z_score'] , np.mean(accuracy_result['AutoTSK_with_Z_score']))

