import numpy as np
from sklearn import meab_squared_error
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['Id'], axis = 1)
dataset_num = dataset.select_dtypes(include = ['float64', 'int64'])

def split_dataset(dataset, test_size = 0.25):
    test_indeces = np.random.rand(len(dataset)) < test_size
    return dataset[~test_indeces], dataset[test_indeces]

train_ds_pd, valid_ds_pd = split_dataset(dataset, test_size = 0.25)
print(f"Example of training: {train_ds_pd.shape}, Example of validation: {valid_ds_pd}")
"""
Example of training: (1107, 80), Example of validation: (353, 80)
"""
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label = label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label = label, task = tfdf.keras.Task.REGRESSION)

tuner = tfdf.tuner.RandomSearch(num_trials = 15, use_predefined_hps = True)
tuned_model = tfdf.keras.GradientBoostedTreesModel(tuner = tuner, task = tfdf.keras.Task.REGRESSION)
tuned_model.fit(train_ds, validation_data = valid_ds, verbose = 2)

tuning_logs = tuned_model.make_inspector().tuning_logs()
best_params = tuning_logs[tuning_logs.best].iloc[0]
print(best_params)

"""
score                                           -28768.640625
evaluation_time                                    470.986292
best                                                     True
split_axis                                     SPARSE_OBLIQUE
sparse_oblique_projection_density_factor                  5.0
sparse_oblique_normalization                          MIN_MAX
sparse_oblique_weights                             CONTINUOUS
categorical_algorithm                                  RANDOM
growing_strategy                            BEST_FIRST_GLOBAL
max_num_nodes                                           128.0
sampling_method                                        RANDOM
subsample                                                 0.9
shrinkage                                                0.05
min_examples                                               20
num_candidate_attributes_ratio                            0.9
max_depth                                                 NaN
Name: 9, dtype: object
"""
best_model = tfdf.keras.GradientBoostedTreesModel(
    task=tfdf.keras.Task.REGRESSION,           
    split_axis="SPARSE_OBLIQUE",            
    sparse_oblique_projection_density_factor=5.0, 
    sparse_oblique_normalization="MIN_MAX",     
    sparse_oblique_weights="CONTINUOUS",       
    categorical_algorithm="RANDOM",           
    growing_strategy="BEST_FIRST_GLOBAL",       
    max_num_nodes=128.0,                      
    sampling_method="RANDOM",             
    subsample=0.9,                      
    shrinkage=0.05,              
    min_examples=20,                             
    num_candidate_attributes_ratio=0.9
)

best_model.compile(metrics = ['MSE'])
best_model.fit(train_ds, validation_data = valid_ds, verbose = 5)
evaluation = best_model.evaluate(valid_ds)

y_predicted = best_model.predict(valid_ds)
y_truee = valid_ds_pd['SalePrice'].values
rmse = np.sqrt(mean_squared_error(np.log(y_truee), np.log(y_predicted)))
print(rmse)
"""
0.14408254997535644
"""
