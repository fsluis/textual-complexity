import pandas as pd

'''
# Run using:
source activate jupyter-tensorflow
export PYTHONPATH=/com1-github/python
python scripts/prepare_combis.py
'''

import os
data_dir = "data/feature_sets/unused/"
print("data_dir:", data_dir)

from helpers.load_parquet_data import load_parquet_data

from itertools import combinations

def all_combinations(feature_idxs, depth, auto_interactions=False):
    # depth=2 -> [2], depth=3 -> [2,3]
    # original features
    all_combis = list()
    all_combis.append( [ [i] for i in feature_idxs] )
    if(auto_interactions):
        all_combis.append( [ (i,i) for i in feature_idxs ] )
    # two- and three-way interactions
    c = [list(combinations(feature_idxs, d)) for d in range(2, depth+1)]
    all_combis.extend(c)
    return all_combis


import torch
import numpy as np
import scipy.stats as stats


def f_classif_gpu(X, y):
    classes = torch.unique(y)
    n_classes = classes.size(0)
    n_features = X.size(1)

    # Calculate class-wise statistics
    class_means = torch.zeros(n_classes, n_features, device=X.device)
    class_variances = torch.zeros(n_classes, n_features, device=X.device)
    class_sample_counts = torch.zeros(n_classes, dtype=torch.float, device=X.device)

    for i, c in enumerate(classes):
        X_class = X[y == c]
        class_means[i] = torch.mean(X_class, dim=0)
        class_variances[i] = torch.var(X_class, dim=0)
        class_sample_counts[i] = X_class.size(0)

    # Calculate overall statistics
    overall_mean = torch.mean(X, dim=0)
    overall_variance = torch.var(X, dim=0)
    overall_sample_count = X.size(0)

    # Calculate between-class sum of squares (SSB)
    SSB = torch.sum(
        class_sample_counts * (class_means - overall_mean) ** 2,
        dim=0
    )

    # Calculate within-class sum of squares (SSW)
    SSW = torch.sum(
        (class_sample_counts - 1) * class_variances,
        dim=0
    )

    # Calculate F-statistics
    F_values = (SSB / (n_classes - 1)) / (SSW / (overall_sample_count - n_classes))

    # Calculate p-values
    df1 = n_classes - 1
    df2 = overall_sample_count - n_classes
    p_values = stats.f.sf(F_values.cpu().numpy(), df1, df2)

    return F_values, p_values


def calc_combi_gpu(X, combi):
    # if isinstance(combi, int):
    if len(combi) == 1:
        x = X[:, combi]
    elif len(combi) == 2:
        x = X[:, combi[0]] * X[:, combi[1]]
    elif len(combi) == 3:
        x = X[:, combi[0]] * X[:, combi[1]] * X[:, combi[2]]
    return x


from tqdm import tqdm


def f_test_gpu(X, y, all_combis):
    all_combis_flat = [combi for combis in all_combis for combi in combis]
    out_features = len(all_combis_flat)
    fs = np.zeros([out_features, 2])
    cs = []

    with tqdm(total=out_features, desc='Processing combinations') as pbar:
        for i, combi in enumerate(all_combis_flat):
            x = calc_combi_gpu(X, combi)
            fp = f_classif_gpu(x.reshape(-1, 1), y)
            fs[i, 0] = fp[0][0]  # f value, first value (first class?)
            fs[i, 1] = fp[1][1]  # p value
            cs.append(combi)
            pbar.update(1)

    return pd.DataFrame({'combi': cs, 'f': fs[:, 0], 'p': fs[:, 1]})


import concurrent.futures


def process_combi(combi, X, y):
    x = calc_combi_gpu(X, combi)
    fp = f_classif_gpu(x.reshape(-1, 1), y)
    return fp[0][0], fp[1][1], combi


def f_test_gpu_concurrent(X, y, all_combis, num_threads):
    all_combis_flat = [combi for combis in all_combis for combi in combis]
    out_features = len(all_combis_flat)
    fs = np.zeros([out_features, 2])
    cs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor, \
            tqdm(total=out_features, desc='Processing combinations') as pbar:
        futures = []
        for combi in all_combis_flat:
            future = executor.submit(process_combi, combi, X, y)
            futures.append(future)

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            f_value, p_value, combi = future.result()
            fs[i, 0] = f_value
            fs[i, 1] = p_value
            cs.append(combi)
            pbar.update(1)

    return pd.DataFrame({'combi': cs, 'f': fs[:, 0], 'p': fs[:, 1]})


import os
import re
from sklearn.preprocessing import StandardScaler
import pyarrow as pa
import pyarrow.parquet as pq

data_files = list()
for file in os.listdir(data_dir):
    # check only text files
    if file.endswith('.parquet') and not "combi" in file:
        data_files.append(file)
#data_files = ["wundt_v16_8-model-ridge-na.parquet"]

max_features = 100000
for data_file in data_files:
    for depth in [2,3]:
        combis_file = data_dir + re.sub('.parquet$', '-combis_depth'+str(depth)+'.parquet', data_file)
        print(combis_file)
        if not os.path.exists(combis_file):
            X, y, z, X_guardian, y_guardian,features = load_parquet_data(data_file, data_dir)

            feature_idxs = [X.columns.get_loc(c) for c in features]
            all_combis = all_combinations(feature_idxs, depth)
            print("#combis:", [len(c) for c in all_combis] )

            # scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_gpu = torch.tensor(X_scaled).to("cuda:0")
            y_gpu = torch.tensor(y.to_numpy()).to("cuda:0")
            print("X_gpu.shape:",X_gpu.shape)

            sorted_combis = f_test_gpu(X_gpu, y_gpu, all_combis).\
                sort_values('f', ascending=False).\
                head(max_features)
            print("sorted_combis.shape:",sorted_combis.shape)

            # Convert the combined DataFrame to a PyArrow table
            table = pa.Table.from_pandas(sorted_combis)

            # Write the table to a Parquet file
            pq.write_table(table, combis_file)
