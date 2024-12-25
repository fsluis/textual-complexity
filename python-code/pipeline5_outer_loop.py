
#%%
import logging
PARAMS = {
    'docker': False, # Use in Docker (meaning logging output gets redicted to stdout)
    'gpu': True, # forces gpu / cpu, error if not available
    'outer_table_name': 'pytorch_cv_step_4_v5',
    'inner_table_name': 'pytorch_cv_step_4_v5_inner',
    'validate_every': 10,
    'batch_size': 128,
    'stopper_patience': 50, # should be at multiple of validate_every (the min is taken from stopper_patience / validate_every # loss samples.
    'stopper_delta': 0,
    'stopper_min_epochs': 20,
    'epoch': 10,
    'savefig': True,
    'net_logging_level': logging.INFO,
    'db_host': 'XXX.XXX.XXX.XXX:XXXX'
}

#%%
import pandas as pd
import pyarrow
import torch
import tqdm
import numpy as np
import torch
import os
import sys

rng = np.random.RandomState(313)

data_dir = "feature_sets/"
output_dir = "logs/torch_cv/"
# Note: Expect those paths relative to working directory


if(PARAMS['gpu']):
    device = 'cuda:0' if torch.cuda.is_available() else sys.exit('No GPU available')
else:
    device = 'cpu'

torch.manual_seed(313)
print(device)

#%%
import logging

# Used to build a log for the individual grid rows / classification attemps
def new_log(name, filename):
    log = logging.getLogger(name)
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    #log.setLevel(logging.WARNING)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(output_dir + filename,mode='w') # w means overwrite?
    file_handler.setFormatter(format)
    log.addHandler(file_handler)
    log.setLevel(PARAMS['net_logging_level'])
    log.propagate = False
    return log

log = new_log("test", "test.log")
log.setLevel(logging.INFO)
log.info("test")
#log.shutdown()
log.handlers.clear()


#%%
# Setting up a root logger for jupyter (otherwise it doesn't work)
# From https://stackoverflow.com/questions/54246623/how-to-toggle-jupyter-notebook-display-logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(format=' %(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("test info")

#%%
from helpers.load_parquet_data import load_parquet_data
data_file = "wundt_v16_8-model-ridge-na.parquet"
X, y, z, X_guardian, y_guardian,features = load_parquet_data(data_file)
y_guardian
#%%
import torch
import numpy as np

X_ts = torch.from_numpy(X.to_numpy(np.float32)).to(device)
y_ts = torch.from_numpy(y.to_numpy(np.int64)).to(device)
X_guardian_ts = torch.from_numpy(X_guardian.to_numpy(np.float32)).to(device)
X_ts.shape
#%%
import torch

def random_undersample_t(Xt, yt, z=None):
    class_counts = torch.bincount(yt)
    min_class_count = torch.min(class_counts)
    #
    sampled_indices = []
    for class_label in torch.unique(yt):
        indices = torch.where(yt == class_label)[0]
        sampled_indices.append(indices[torch.randperm(len(indices))[:min_class_count]])
    #
    sampled_indices = torch.cat(sampled_indices)
    sampled_X = Xt[sampled_indices]
    sampled_y = yt[sampled_indices]
    sampled_z = None
    if not z is None:
        sampled_z = z.iloc[sampled_indices.detach().cpu().numpy()]
    #
    return sampled_X, sampled_y, sampled_z, sampled_indices

# Example usage:
X_artificial = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_artificial = torch.tensor([0, 1, 0, 1, 0])

sampled_X, sampled_y, sampled_z, sampled_indices = random_undersample_t(X_artificial, y_artificial)

print("Sampled X:")
print(sampled_X)

print("Sampled y:")
print(sampled_y)

print("Sampled idxs:")
print(sampled_indices)

#%%

def standardize_t(tensor, mean=None, std=None):
    if(mean==None): mean = tensor.mean(dim=0)
    if(std==None): std = tensor.std(dim=0)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor, mean, std


#%%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def confusion_matrix_scorer(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1], 'fn': cm[1, 0], 'tp': cm[1, 1]}

def metrics(y_true, y_pred):
    return {
        'classif.acc': accuracy_score(y_true,y_pred),
        'classif.precision': precision_score(y_true,y_pred),
        'classif.recall': recall_score(y_true,y_pred),
        'classif.f1': f1_score(y_true,y_pred),
        'classif.tn': confusion_matrix_scorer(y_true,y_pred)['tn'],
        'classif.fp': confusion_matrix_scorer(y_true,y_pred)['fp'],
        'classif.fn': confusion_matrix_scorer(y_true,y_pred)['fn'],
        'classif.tp': confusion_matrix_scorer(y_true,y_pred)['tp'],
    }
#%%
import torch
import torch.nn as nn
from helpers.early_stopper import EarlyStopper

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        #x = self.sigmoid(x)
        return x

from com1py.net import LogitNet
model = LogisticRegression(25).to(device)
net = LogitNet(
    model=model,
    criterion=torch.nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam(model.parameters(),lr=.001),
    scheduler=None,
    stopper=EarlyStopper(patience=10, min_delta=.0001),
    validate_every=10,
    batch_size=128
)
net
#%%
data_files = list()
for file in os.listdir(data_dir):
    # check only text files
    if file.endswith('.parquet') and not "combi" in file:
        data_files.append(file)
#data_files = ["wundt_v16_8-model-ridge-na.parquet"]

data_files = pd.Series(data_files)

#%%
from helpers.load_parquet_data import load_parquet_data
data_file = "wundt_v16_8-model-ridge-na.parquet"
X, y, z, X_guardian, y_guardian,features = load_parquet_data(data_file)
y_guardian
#%%
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

# train-test split
outer_cv = GroupShuffleSplit(n_splits=1, random_state=rng, test_size=None, train_size=0.8)

# Manual loop
for train_idxs, test_idxs in outer_cv.split(X_ts, y_ts, groups=z):
    X_train, y_train = X_ts[train_idxs], y_ts[train_idxs]
    X_test, y_test = X_ts[test_idxs], y_ts[test_idxs]

    start_time = time.time()
    net.train(PARAMS["epoch"], X_train, y_train.float(), X_test, y_test.float())
    log.info("--- %s seconds ---" % (time.time() - start_time))

    logit, prob, pred = net.predict(X_test)
    # This somehow gives a weird numpy bug in cpu mode
    m = metrics(y_test.detach().cpu(), pred.detach().cpu())
    log.info("metrics:%s" % m)

    # Plot the training loss
    #fig = net.plot_losses()
    #plt.show()


#%% md
# # Grid
# - The net__ part of the param names gets separated in the inner loop
# - List values are passed onto the inner loop for tuning
#%%
from itertools import product
import pandas as pd

def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())],
                        columns=dictionary.keys())

wundt_files = data_files[data_files.str.contains('wundt')]
wundt_features = 25
wundt_grid = expand_grid({
    'data_file': wundt_files,
    'outer_epoch': [1000],
    'inner_epoch': [100],
    'net__depth': [1,2,3],
    'net__width': list(range(4, 164+1, 4)), # 40 values,
    'net__activation': [['sigmoid', 'tanh', 'relu', 'softmax']], # 'identity',
    'net__dropout': [[0,.1,.2,.3,.4]], #
    'net__lr_rate': [[.00001, .0001, .001, .01, .1]], # higher values (0.2, 0.3) don't work
    'net__weight_decay': [[.0001, .001, .01, .1]] # Recommendation is 1e-4 to 1e-1. GlmNET uses as minimum .0001. Shouldn't be too low, as we want some level of regularlization.
})

wundt_grid = wundt_grid.loc[wundt_grid.astype(str).drop_duplicates().index].reset_index(drop=True) # drop drops the old index
wundt_grid
#%%
bert_files = data_files[data_files.str.contains('wundt')==False]
bert_features = 768
bert_grid = expand_grid({
    'data_file': bert_files,
    'outer_epoch': [1000],
    'inner_epoch': [100],
    'net__depth': [1,2,3],
    'net__width': list(range(4, 164+1, 4)), # 40 values
    'net__activation': [['sigmoid', 'tanh', 'relu', 'softmax']], # 'identity',
    'net__dropout': [[0,.1,.2,.3,.4]], # 0,
    'net__lr_rate': [[.00001, .0001, .001, .01, .1]], # higher values (0.2, 0.3) don't work
    'net__weight_decay': [[.0001, .001, .01, .1]] # Recommendation is 1e-4 to 1e-1. GlmNET uses as minimum .0001. Shouldn't be too low,
})
bert_grid
#%%
#outer_grid = wundt_grid
#outer_grid = bert_grid
bert_grid.loc[bert_grid['net__depth']==1, ['net__width', 'net__dropout', 'net__activation']] = None
bert_grid_dedup = bert_grid.loc[bert_grid.astype(str).drop_duplicates(keep='first').index]
wundt_grid.loc[wundt_grid['net__depth']==1, ['net__width', 'net__dropout', 'net__activation']] = None
wundt_grid_dedup = wundt_grid.loc[wundt_grid.astype(str).drop_duplicates(keep='first').index]
outer_grid = pd.concat([wundt_grid_dedup,bert_grid_dedup])
outer_grid.shape
#%%

row = outer_grid.iloc[:1,:]
column_names = "(`" + "`,`".join(row.keys()) + "`)"
values = "('" + "','".join(row.astype(str)) + "')"
query = f"INSERT INTO xxx {column_names} VALUES {values}"

import math

row = outer_grid.iloc[0,:].to_dict()
values = [str(v) for v in row.values()]
values = [ "NULL" if v.lower() == "nan" or v.lower()=="none" else "'"+v+"'" for v in values ]
assignment_list = ", ".join(["`"+k+"`="+v for k,v in zip(row.keys(), values)])
pk_value = 1
query = f"UPDATE {PARAMS['outer_table_name']} SET {assignment_list} WHERE id = {pk_value}"

query

#%% md
# # MySQL
#%%
#!pip install sqlalchemy
#!pip install mysql-connector-python
#!pip install mysql-python < doesn't work, use pure-python version
#!pip install pymysql
#!pip install sqlalchemy_utils
#%%

double_columns = ['tuned_dropout', 'tuned_lr_rate', 'tuned_weight_decay', 'classif.acc', 'classif.precision',
                  'classif.recall', 'classif.f1', 'classif.tn', 'classif.fp',
                  'classif.fn', 'classif.tp', 'guardian.rho_noco2',
                  'guardian.rlogit_noco2', 'guardian.rprob_noco2', 'guardian.rho_comp',
                  'guardian.rlogit_comp', 'guardian.rprob_comp', 'guardian.rho_pf',
                  'guardian.rlogit_pf', 'guardian.rprob_pf','avg_train_loss','test_loss']
double_dict = {k:pd.Series(dtype='double') for k in double_columns}
int_dict = {k:pd.Series(dtype='int') for k in ['early_stopping', 'total_epochs', 'best_epoch']}
string_dict = {k:pd.Series(dtype='string') for k in ['tuned_activation']}

result_df = pd.DataFrame({**int_dict,**string_dict,**double_dict}) #, index=range(wundt_grid.shape[0])
temp_df = pd.merge(outer_grid,result_df,how='outer',left_index=True,right_index=True)
temp_df
# can be used to create mysql table
temp_df.columns
#%%
from sqlalchemy import create_engine
from sqlalchemy import text
import sqlalchemy
# timeouts: https://stackoverflow.com/questions/29755228/sqlalchemy-mysql-lost-connection-to-mysql-server-during-query
engine = create_engine(f"mysql+pymysql://XXXXX:XXXXXXXXX@{PARAMS['db_host']}/com1_torch", isolation_level="AUTOCOMMIT", pool_recycle=3600, pool_pre_ping=True )
# AUTOCOMMIT flag means statements are committed upon calling execute

# Create outer table
def create_table(temp_df, table_name, foreign_key_column = None):
    if not sqlalchemy.inspect(engine).has_table(table_name):
        with engine.connect() as con:
            df_empty = temp_df.head(0)
            df_empty.to_sql(con=con, name=table_name, if_exists='fail', index=False)
            con.execute(text(f'ALTER TABLE {table_name} ADD id BIGINT PRIMARY KEY AUTO_INCREMENT;'))
            if foreign_key_column:
                con.execute(text(f'ALTER TABLE {table_name} ADD {foreign_key_column} BIGINT;'))  # Add foreign key column
            con.close()
            logger.info(f"Created table {table_name}")

create_table(temp_df, PARAMS['outer_table_name'])

# Create inner table
inner_columns = [col for col in temp_df.columns if col.startswith("classif") or col.startswith("net__") or col.endswith("_loss") ]
create_table(temp_df[inner_columns], PARAMS['inner_table_name'], "outer_id")

# Fetch the table data from the MySQL database
query = f"SELECT * FROM {PARAMS['outer_table_name']}"
with engine.connect() as con:
    sql_data = pd.read_sql_query(text(query), con) #.drop("id", axis=1)

print('before: ',outer_grid.shape)
before_grid = outer_grid.copy()

# From https://stackoverflow.com/questions/25104259/how-to-properly-escape-strings-when-manually-building-sql-queries-in-sqlalchemy
#from sqlalchemy_utils import escape_like
def escape_like(l):
    return l.replace('\'','\"')

# Helper function that check if row exists
def mysql_row_id(engine, pd_row, logger = None, table_name = PARAMS['outer_table_name']):
    with engine.connect() as con:
        values = [ "NULL" if v.lower() == "nan" or v.lower()=="none" else "'"+v+"'" for v in [escape_like(str(v)) for v in pd_row]] # dict_row.astype(str)
        # Create the WHERE statement
        where_statement = " AND ".join([f"`{name}` = {value}" for name, value in zip(pd_row.keys(), values)  if value!="NULL" ])
        #
        # Create the WHERE query
        where_query = f"SELECT `id` FROM {table_name} WHERE {where_statement} LIMIT 1"
        #
        # Execute the query and fetch the result
        if(logger):
            logger.info("mysql_row_exists: " + where_query)
        result = con.execute(text(where_query)).fetchone()
        con.close()
        #
        # Return the ID if found, otherwise None
        return result[0] if result else None


# row is a pd series
# returns primary key auto increment value
def mysql_insert_row(engine, dict_row, logger = None, table_name = PARAMS['outer_table_name']):
    with engine.connect() as con:
        # Create the INSERT statement with column names
        column_names = "(`" + "`,`".join(dict_row.keys()) + "`)"
        values = [ "NULL" if v.lower() == "nan" or v.lower()=="none" else "'"+v+"'" for v in [escape_like(str(v)) for v in dict_row.values()]] # dict_row.astype(str)
        values = "(" + ",".join(values) + ")"
        query = f"INSERT INTO {table_name} {column_names} VALUES {values}"
        #print(query)
        if(logger):
            logger.info("mysql_insert_row: " + query)
        con.execute(text(query))

        # Fetch the primary key value
        pk_value = con.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]
        con.close()
        return pk_value

# here, row is a dict!
def mysql_update_row(engine, dict_row, pk_value, logger, table_name = PARAMS['outer_table_name']):
    with engine.connect() as con:
        values = [ "NULL" if v.lower() == "nan" or v.lower()=="none" else "'"+v+"'" for v in [escape_like(str(v)) for v in dict_row.values()]] # dict_row.astype(str)
        assignment_list = ", ".join(["`" + k +"`=" + v for k,v in zip(dict_row.keys(), values) if v!="NULL"])
        query = f"UPDATE {table_name} SET {assignment_list} WHERE id = {pk_value}"
        #print(query)
        logger.info("mysql_update_row: " + query)
        con.execute(text(query))
        con.close()

#print(mysql_row_id(engine, outer_grid.iloc[:1, :], logger))

outer_grid = before_grid # don't want to pre-empt the grid now, better redo / finish the early ones



#%%
# # Net Builder
#%%
from com1py.net import LogitNet
from com1py.torch_models import LogisticRegression as Model1
from com1py.torch_models import Model2, Model3
import torch

def build_net(in_features, out_features, logger=logging.getLogger(), depth=1, lr_rate=.001, weight_decay=0,  width=25, activation=torch.nn.Tanh(),  dropout=.2, **kwargs):
    #print("Unused params:", kwargs)
    assert len(kwargs)==0, "unused params encountered while building Net"+str(kwargs)

    if activation=="sigmoid":
        fun = torch.nn.Sigmoid()
    elif activation=="tanh":
        fun = torch.nn.Tanh()
    elif activation=="identity":
        fun = torch.nn.Identity()
    elif activation=="relu":
        fun = torch.nn.ReLU()
    elif activation=="softmax":
        fun = torch.nn.Softmax()
    elif activation==None:
        fun = None # only for depth=1 models
    else:
        log.error("Unknown activation function: %s" % activation)

    if depth==1:
        model = Model1(in_features, out_features).to(device)
    elif depth==2:
        #print(in_features, out_features, width, fun)
        model = Model2(in_features, out_features, int(width), fun, dropout).to(device)
    elif depth==3:
        model = Model3(in_features, out_features, int(width), fun, dropout).to(device)
    else:
        log.error("Unknown depth value encountered %d:" % depth)
        return

    net = LogitNet(
        model=model,
        criterion=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(model.parameters(),lr=lr_rate,weight_decay=weight_decay),
        scheduler=None,
        stopper=EarlyStopper(patience=PARAMS['stopper_patience'], min_delta=PARAMS['stopper_delta'], min_epochs=PARAMS['stopper_min_epochs'], logger=logger),
        validate_every=PARAMS['validate_every'],
        batch_size=PARAMS['batch_size'],
        logging_level=PARAMS['net_logging_level'],
        logger=logger
    )

    return net

#%% md
# # Train loop
#%%
import time

def preprocess(X_ts, y_ts, z, train_idxs, test_idxs):
    Xt_train, yt_train, z_train, _ = random_undersample_t(X_ts[train_idxs], y_ts[train_idxs], z.iloc[train_idxs])
    # no need to undersample the test observations
    Xt_test, yt_test = X_ts[test_idxs,:], y_ts[test_idxs]
    return Xt_train, yt_train, Xt_test, yt_test, z_train


def preprocess_and_standardize(X_ts, y_ts, z, train_idxs, test_idxs):
    Xt_train, yt_train, z_train, _ = random_undersample_t(X_ts[train_idxs], y_ts[train_idxs], z.iloc[train_idxs])
    # no need to undersample the test observations
    # Standardizing test data using distribution from the train data
    Xt_train, train_mean, train_std = standardize_t(Xt_train)
    Xt_test, _, _ = standardize_t(X_ts[test_idxs,:], train_mean, train_std) # apply same dist to test set
    yt_test = y_ts[test_idxs]
    #Xt_test, yt_test = X_ts[test_idxs,:], y_ts[test_idxs]
    #
    return Xt_train, yt_train, Xt_test, yt_test, z_train, train_mean, train_std

def train_net(Xt_train, yt_train, Xt_test, yt_test, net_params, epoch, logger):
    log.info("Train net params: %s" % net_params)
    net = build_net(Xt_train.shape[1], 1, **net_params, logger=logger)

    if not yt_test is None:
        yt_test = yt_test.float()

    start_time = time.time()
    train_loss, test_loss = net.train(epoch, Xt_train, yt_train.float(), Xt_test, yt_test)
    log.info("--- %s seconds ---" % (time.time() - start_time))

    return net, train_loss, test_loss

# dict_b replaces dict_a
def extract_net_params(dict_a, dict_b):
    all_params = {**dict_a, **dict_b}
    net_params = {k[5:]: v for k, v in all_params.items() if k.startswith('net__')}
    return net_params, all_params

#%% md
# # Inner loop
#%%
from torch import FloatTensor
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
import copy
import time
from tqdm import tqdm

criterion = torch.nn.BCEWithLogitsLoss()

#%%

def inner_loop_from_mysql(outer_row, outer_id, log):
    log.info("Inner loop cross-validation")

    # Extract nested params
    cols = [c for c in outer_row.index if isinstance(outer_row[c], list)]
    inner_grid = expand_grid(outer_row[cols].to_dict())
    
    # Check if there's something to tune
    if inner_grid.shape[0] < 2:
        log.info("Skipping inner loop cause there is only %d rows to test" % inner_grid.shape[0])
        return None,None,None # no params to optimize in an inner loop

    # Check if the MySQL table contains all inner rows
    with engine.connect() as con:
        total_rows_in_db = con.execute(text(f"SELECT COUNT(*) FROM {PARAMS['inner_table_name']}"
                                            f" WHERE outer_id = :outer_id  AND `classif.acc` IS NOT NULL"), {'outer_id': outer_id}).scalar()

    if not total_rows_in_db >= inner_grid.shape[0]: # some have duplicates / simultaneous runs it seems - delete later
        log.info("Insufficient rows in inner loop table %d vs. %d" % (inner_grid.shape[0], total_rows_in_db))
        return None,None,None # no params to optimize in an inner loop

    log.info(f"All inner rows already processed for outer_id: {outer_id}. Fetching results from the database.")
    with engine.connect() as con:
        sql_query = f"SELECT * FROM {PARAMS['inner_table_name']} WHERE outer_id = :outer_id AND `classif.acc` IS NOT NULL"
        inner_result = pd.read_sql_query(text(sql_query), con, params={'outer_id': outer_id})

        inner_result.sort_values("test_loss", inplace=True)
        log.info("cross-validation result: %s" % inner_result)

        winning_params = {k: v for k, v in inner_result.iloc[0,:].items() if k.startswith('net__')}
        tune_params = {k: v for k, v in inner_result.iloc[0,:].items() if k in cols}
        log.info("winning params: %s" % winning_params)
        log.info("tuned params: %s" % tune_params)

        return winning_params, tune_params, inner_result

## Little conversion cause some of the params are stored as strings not numbers
def convert_winning_params(winning_params):
    # Define the expected types for each parameter
    expected_types = {
        'net__depth': int,
        'net__width': int,
        'net__activation': str,  # Activation functions are strings
        'net__dropout': float,
        'net__lr_rate': float,
        'net__weight_decay': float,
    }

    # Convert parameters to their expected types
    for param, expected_type in expected_types.items():
        if param in winning_params and winning_params[param] is not None:
            try:
                winning_params[param] = expected_type(winning_params[param])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {param}={winning_params[param]} to {expected_type}.")
                winning_params[param] = None  # Set to None if conversion fails
    return winning_params


#%% md
# # Outer loop
#%%
from sklearn.model_selection import GroupShuffleSplit
from helpers.load_parquet_data import load_parquet_data
from tqdm import tqdm
import gc

def outer_loop(param_grid):
    #outer_results = list()
    # shuffle grid
    param_grid = param_grid.sample(frac=1).reset_index(drop=True)

    counter = 0
    with tqdm(total=param_grid.shape[0]) as pbar:
        for grid_index, grid_row in param_grid.iterrows():
            pbar.update(1)

            # Check if row exists, if not insert and get primary key value.
            pk_value = mysql_row_id(engine, grid_row, logger)
            if(pk_value is None):
                logger.warning(f"Inner loop not run for grid index {grid_index}")
                continue # stops this iteration, continues with rest
            pbar.set_description("ID %d" % pk_value)

            # Check if database row contains final results
            db_row = mysql_row_by_id(engine, pk_value, logger)
            if db_row.get('classif.acc') is not None:
                logger.info("Skipping outer loop for primary key {pk_value}, final results already available.")
                continue

            log = new_log(f"row{pk_value}", f"row{pk_value}.txt")
            #log = logger
            log.info("\n\n\n\nOuter loop row index: %d/%d" % (grid_index,param_grid.shape[0]))
            log.info("Outer loop params: %s" % grid_row)

            # Get tuning results
            winning_params, tune_params, tune_result = inner_loop_from_mysql(grid_row, pk_value, log)
            if(winning_params is None):
                log.warning(f"Inner loop not finished for primary key {pk_value}")
                continue  # stops this iteration, continues with rest

            # Load data
            X, y, z, X_guardian, y_guardian,features = load_parquet_data(grid_row['data_file'])
            X_ts = torch.from_numpy(X.to_numpy(np.float32)).to(device)
            y_ts = torch.from_numpy(y.to_numpy(np.int64)).to(device)
            X_guardian_ts = torch.from_numpy(X_guardian.to_numpy(np.float32)).to(device)
            log.info("Loaded X: %s" % str(X_ts.shape))

            # train-test split
            outer_cv = GroupShuffleSplit(n_splits=1, random_state=rng, test_size=None, train_size=0.8)

            # Manual loop
            for train_idxs, test_idxs in outer_cv.split(X, y, groups=z):
                # Preprocess data
                train_xt, train_yt, test_xt, test_yt, train_z, train_mean, train_std = preprocess_and_standardize(X_ts, y_ts, z, train_idxs, test_idxs)

                # Re-train with winning params
                tune_params = convert_winning_params(tune_params)
                net_params, params = extract_net_params(grid_row.to_dict(), tune_params)
                log.info("final net params: %s" % net_params)
                #print("winning_params: %s" % winning_params)
                #print("final net params: %s" % net_params)
                net, train_loss, test_loss = train_net(train_xt, train_yt, test_xt, test_yt, net_params, params['outer_epoch'], log)
                plt = net.plot_losses()
                if PARAMS['savefig']:
                    plt.savefig("%s%d.png" % (output_dir,pk_value) )
                    plt.close()
                #plt.show()
                total_epochs = len(net.train_losses) # number of epochs done
                early_stopping = net.stop_at_
                best_epoch = net.best_epoch_

                # Evaluate model performance using tuned parameters
                logit_ts, prob_ts, pred_ts = net.predict(test_xt)
                pred = pred_ts.detach().squeeze(1).cpu().numpy()
                # calculate test losses
                #print(pred.shape)

                m = metrics(test_yt.detach().cpu().numpy(), pred)
                log.info("train-test result: %s" % m)

            # Re-fit to whole data set (unless there was an early stop)
            if net.stop_at_ is None:
                # only if previous train didn't terminate early
                # Preprocess full data set
                train_idxs = np.arange(len(X_ts))
                test_idxs = np.empty(0)
                train_xt, train_yt, test_xt, test_yt, train_z, train_mean, train_std = preprocess_and_standardize(X_ts, y_ts, z, train_idxs, test_idxs)
                net, train_loss, _ = train_net(train_xt, train_yt, None, None, net_params, params['outer_epoch'], log)
                log.info("Reffited on %d train samples (against zero == %d test samples) with train loss: %f" % (train_xt.size(0), test_xt.size(0), train_loss))
            else:
                log.info("Skipped re-fitting because earlier fit stopped early. Re-using earlier fit for target predictions.")

            # Evaluate on The Guardian
            xt_guardian, _, _ = standardize_t(X_guardian_ts, train_mean, train_std)
            target_eval = guardian_eval(net, xt_guardian, y_guardian, log)
            log.info("Target results: %s" % target_eval)

            #row = {**params, **m, **target_eval}
            tuned_params = {("tuned_"+k[5:]): v for k, v in tune_params.items() }
            row = {**grid_row.to_dict(), **tuned_params, **m, **target_eval}
            row['total_epochs'] = total_epochs
            row['best_epoch'] = best_epoch
            row['early_stopping'] = early_stopping
            #outer_results.append(row)
            mysql_update_row(engine, row, pk_value, log)

            # we seem to be having a mem leak on the cpu ... so something stays glued into mem...
            log.handlers.clear()
            del X,y,z,X_guardian, y_guardian,features,X_ts,y_ts,X_guardian_ts, net,row,target_eval,train_loss,params,m,total_epochs,best_epoch,early_stopping,net_params,log,pred,logit_ts,prob_ts,pred_ts,plt,train_xt, train_yt, test_xt, test_yt, train_z, test_loss, tune_params, tune_result, train_idxs, test_idxs, outer_cv, pk_value,grid_index, grid_row, xt_guardian
            gc.collect()

            counter+=1
    return counter

count = outer_loop(outer_grid)


#results = \
#only_1 = outer_grid.loc[outer_grid['net__depth']==1,]
#count = outer_loop(only_1)

print("\n\nDone\n\n")

