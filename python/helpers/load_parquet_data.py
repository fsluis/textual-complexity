import os
import pandas as pd
default_data_dir = "data/feature_sets/"

def load_parquet_data(data_file, data_dir=default_data_dir):
    # Write the table to a Parquet file
    df = pd.read_parquet(data_dir + data_file)

    # Select columns that start with 'd_' using regex and add 'label' column
    features = df.filter(regex='^d_\d+$').columns
    if features.empty:  # if not d_, then wundt features
        features = list(set(df.columns.tolist()) - set(['label', 'id', 'pair_id', 'lang']))
    features

    # Separate The Guardian data
    df_guardian = pd.read_csv('data/guardian/guardian_stimulus.csv')
    X_guardian = df[df['label'] == 2]
    idxs = df_guardian['stimulus'].tolist()
    # print(X_guardian)
    X_guardian = X_guardian.set_index('id')
    df_guardian = df_guardian.set_index('stimulus')
    # y_guardian = df_guardian.loc[X_guardian['id'], 'noco2']
    X_guardian.rename(index={6064: 6046}, inplace=True) # typo
    guardian_cols = ['noco2','comp','pf','inte']
    X_guardian.loc[idxs, guardian_cols] = df_guardian.loc[idxs, guardian_cols]
    y_guardian = X_guardian[guardian_cols]
    X_guardian = X_guardian[features]
    df = df[df['label'] < 2]

    # clean rows with empty features
    df.dropna(subset=features, how='any', inplace=True)
    X = df[features]
    y = df.label
    z = df.pair_id

    return X, y, z, X_guardian, y_guardian, features
