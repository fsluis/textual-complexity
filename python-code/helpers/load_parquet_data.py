import os
import pandas as pd
default_data_dir = "data/feature_sets/"

def load_parquet_data(data_file, data_dir=default_data_dir):
    # Write the table to a Parquet file
    df = pd.read_parquet(data_dir + data_file)
    # columns: pair_id, lang, [features], label, id
    # features for wundt data: 'causal_word_span_ratio',
    # 'cc_sent_geo_log_p_mean', 'coref_fwin_fill_local_1_mean', 'coref_fwin_fill_local_2_mean',
    # 'coref_fwin_fill_local_3_mean', 'coref_fwin_fill_local_4_mean', 'esa_swe_w30_mean',
    # 'esacoh_fwin_fill_local_1_mean', 'esacoh_fwin_fill_local_2_mean', 'esacoh_fwin_fill_local_3_mean',
    # 'esacoh_fwin_fill_local_4_mean', 'loc__cor2_f0_i0_log_mean_mean', 'lucene_characters_per_word_mean',
    # 'lucene_syllables_per_word_mean', 'noncausal_word_span_ratio', 'snowball_swe_w55_n1_mean',
    # 'snowball_swe_w55_n2_mean', 'snowball_swe_w55_n3_mean', 'snowball_swe_w55_n4_mean', 'snowball_swe_w55_n5_mean',
    # 'wordnet_log_cumulative_spread_1_posall_stonge_mean', 'wordnet_log_cumulative_spread_2_posall_stonge_mean',
    # 'wordnet_log_cumulative_spread_3_posall_stonge_mean', 'wordnet_log_cumulative_spread_4_posall_stonge_mean',
    # 'wordnet_log_cumulative_spread_5_posall_stonge_mean',
    # features for bert data: d_0, d_1, ... d_767
    # label = 0.0 for simple, 1.0 for english, 2.0 for guardian features

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
