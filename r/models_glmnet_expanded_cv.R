
library(plyr) # for rbind.fill function
library("dplyr")
library('DBI')
library(data.table) # %like% operator
library(stringr)

# List data files
data_dir = "data/feature_sets/"
data_files = list.files(data_dir, pattern = "*.parquet")
data_files = data_files[!str_detect(data_files, 'combi')]

# Build experiment grid
outer_grid_depth_1 = expand.grid(  
  data_file = data_files,
  depth = c(1),
  max_features = c(-1),
  alpha = c(1), # 0=ridge, 1=lasso
  findcorr_cutoff = c(.9),
  add_original = c(F) # include all depth=1 variables
)
outer_grid_depth_2_bert = expand.grid(  
  data_file = data_files[!grepl("wundt", data_files)],
  depth = c(2),
  max_features = seq(1000, 30000, 1000),
  alpha = c(1), # 0=ridge, 1=lasso
  findcorr_cutoff = c(.9),
  add_original = c(T) # include all depth=1 variables
)
#
outer_grid_depth_3_bert = expand.grid(  
  data_file = data_files[!grepl("wundt", data_files)],
  depth = c(3),
  max_features = seq(1000, 30000, 1000),
  alpha = c(1), # 0=ridge, 1=lasso
  findcorr_cutoff = c(.9),
  add_original = c(T) # include all depth=1 variables
)
#
# max 326 features for wundt x 2
# max 2626 features for wundt x 3
outer_grid_depth_2_wundt = expand.grid(  
  data_file = data_files[grepl("wundt", data_files)],
  depth = c(2),
  max_features = seq(50,350,10),#seq(50,400,50),
  alpha = c(1), # 0=ridge, 1=lasso
  findcorr_cutoff = c(.9),
  add_original = c(T) # include all depth=1 variables
)
#
outer_grid_depth_3_wundt = expand.grid(  
  data_file = data_files[grepl("wundt", data_files)],
  depth = c(3),
  max_features = seq(100,2700,100),
  alpha = c(1), # 0=ridge, 1=lasso
  findcorr_cutoff = c(.9),
  add_original = c(T) # include all depth=1 variables
)
#
expanded_grid = rbind(outer_grid_depth_1, outer_grid_depth_2_bert, outer_grid_depth_3_bert, outer_grid_depth_2_wundt, outer_grid_depth_3_wundt)
#expanded_grid = outer_grid_depth_1
outer_grid = expanded_grid


## Check which part of grid is already done
  host = "localhost"
  port = 3306
drv = dbDriver("MySQL")
con = dbConnect(drv, user="XXXXX", dbname="com1_ensiwiki-2020_agerank", host=host, port=port, password="XXXXXXXXXX")

# Check if the table already exists in the database
table_name <- "glmnet_cv"
if (!dbExistsTable(con, table_name)) {
  # Add table to database
  dplyr::copy_to(con, expanded_grid, table_name, temporary = FALSE)
  # gives an error but does create an empty table
  
  # Add any additional columns to the table
  extra_columns <- c("features.all", "features.dropped","features.used")
  for (col in extra_columns) {
    dbExecute(con, paste0("ALTER TABLE `", table_name, "` ADD COLUMN `", col, "` INT"))
  }
  extra_columns <- c("classif.acc", "classif.precision", "classif.recall", "classif.fbeta", "classif.ce", "classif.fn", "classif.fp", "classif.tn", "classif.tp", "guardian.rho_noco2", "guardian.rlogit_noco2", "guardian.rprob_noco2", "guardian.rho_comp", "guardian.rlogit_comp", "guardian.rprob_comp", "guardian.rho_pf", "guardian.rlogit_pf", "guardian.rprob_pf", "guardian.rinte", "guardian.r2inte")
  for (col in extra_columns) {
    dbExecute(con, paste0("ALTER TABLE `", table_name, "` ADD COLUMN `", col, "` DOUBLE"))
  }
  dbExecute(con, paste0("ALTER TABLE ", table_name, " ADD COLUMN ID BIGINT AUTO_INCREMENT PRIMARY KEY"))
}
query <- paste0("SELECT ",paste(colnames(expanded_grid), collapse = ", ") ," FROM ", table_name)
sql_data <- dbGetQuery(con, query)

# Identify the rows in the dataframe that are not in the table
outer_grid <- na.omit(expanded_grid[!duplicated(rbind(sql_data, expanded_grid)), ])

# Close the database connection
dbDisconnect(con)

# max 4 gb?
library(parallel)

## Check which part of grid is already done
nr_of_nodes = 3 # 4 if features <= 20000
cat(paste("Using", nr_of_nodes, "cores"))

cluster <- makeCluster(nr_of_nodes)
library(doParallel)
registerDoParallel(cluster)
library(doSNOW)
registerDoSNOW(cluster)


library(progress)
pb <- progress_bar$new(
  format = "combi=:row i=:i of :total [:bar] :elapsed | eta: :eta",
  total = nrow(outer_grid),    # 100 
  width = 100)
progress_letter <- rep(LETTERS[1:10], 10)  # token reported in progress bar
# allowing progress bar to be used in foreach -----------------------------
progress <- function(i){
  pb$tick(tokens = list(row = row_indexes[i], i=i, total=nrow(outer_grid) ))
} 
opts <- list(progress = progress)

row_indexes = sample(1:nrow(outer_grid)) # randomizes the order
#row_indexes = 1:nrow(outer_grid) # don't randomizes the order
cat("Starting row:", row_indexes[1],'\n')

# %dopar%
outer_results = foreach(i=1:length(row_indexes), .packages = c("glmnet", "glmnetUtils", "foreach", "plyr", "dplyr", "DBI","stringr"), .options.snow=opts, .combine='rbind.fill') %dopar% {
  #for (i in row_indexes) {
  row = row_indexes[i]
  drop_na_level = 1
  seed_value = 313
  alpha = outer_grid[row, 'alpha']
  depth = outer_grid[row, 'depth']
  max_features = outer_grid[row, 'max_features']
  findcorr_cutoff = outer_grid[row, 'findcorr_cutoff']
  add_original = outer_grid[row, 'add_original']
  lambda.min.ratio=1/10^(4-depth+1) # 4 is default, n is min. 1
  data_file = outer_grid[row, 'data_file']
  combi_filename = paste0(str_sub(paste0(data_dir, data_file), end=-9),"-combis_depth", depth, ".parquet")
  
  if(depth>1 && !file.exists(combi_filename)) {
    cat(sprintf("Skipping combination %d (i=%d:%d)", row, i, nrow(outer_grid)),' (could not read combi file)\n')
    return(NULL)
  }
  
    library("RMySQL")

  res = as.list(outer_grid[row,])
  res$data_file = as.character(res$data_file)
  res$add_original = as.integer(res$add_original)
  # Insert the row into the MySQL table
  drv = dbDriver("MySQL")
  con = dbConnect(drv, user="XXXXX", dbname="com1_ensiwiki-2020_agerank", host=host, port=port, password="XXXXXXXXXX")
  # Create WHERE statement
  where_statement = paste0(paste(names(res), res, sep = " = '", collapse = "' AND "), "'")
  where_query <- paste0("SELECT COUNT(*) AS total FROM ", table_name, " WHERE ", where_statement)
  count_value <- dbGetQuery(con, where_query)
  
  if(count_value==0) {
    # Create the INSERT statement with column names
    column_names <- paste0("(`", paste(colnames(outer_grid), collapse = "`,`"), "`)")
    values <- paste0("('", paste(res, collapse = "','"), "')")
    query <- paste0("INSERT INTO ", table_name, " ", paste(column_names, collapse = ","), " VALUES ", values)
    dbExecute(con, query)
    # primary key value
    pk_value <- dbGetQuery(con, "SELECT LAST_INSERT_ID()")$`LAST_INSERT_ID()`
    dbDisconnect(con)
  } else {
    cat(sprintf("Skipping combination %d (i=%d:%d)", row, i, nrow(outer_grid)),'\n')
    dbDisconnect(con)
    return(NULL)
  }

  log_filename = paste0("logs/glmnet_cv/glmnet_cv_", pk_value,".txt")
  capture.output({
    tryCatch({ # to make sure any error ends up in the log, and is not lost
      ## Load data
  library(arrow)
  df = read_parquet(paste0(data_dir, data_file))
  df_guardian = df[df$label==2,]
  df_wiki = df[df$label<2,]
  
  ## Features
  label_idx = which(colnames(df_wiki)=='lang')
  label_idxs = c(label_idx, which(colnames(df_wiki)=='pair_id') )
  label_idxs = c(label_idxs, which(colnames(df_wiki)=='id') )
  label_idxs = c(label_idxs, which(colnames(df_wiki)=='label') )
  label_idxs = c(label_idxs, which(colnames(df_wiki)=='length') )
  label_idxs = c(label_idxs, which(colnames(df_wiki)=='text') )
  label_idxs = c(label_idxs, which(colnames(df_wiki)=='wiki_id') )
  label_idxs = c(label_idxs, which(colnames(df_wiki)=='filename') )
  feature_names = colnames(df_wiki)[-label_idxs]
  df_wiki =df_wiki[complete.cases(df_wiki[,feature_names]), ]
  
  library("mlr3")
  #library(tidyverse)
  #library(dplyr)
  library("magrittr")
  library(mlr3verse)
  library(mlr3pipelines)
  library(mlr3filters)
  #remotes::install_github("mlr-org/mlr3pipelines@pipeop_filterrows")
  source("PipeOpFilterRows.R") # copied from github branch ... integrated here.
  source("wundt_tuning_functions.R")
  
  # Needed to limit to 1 core
  library(future)
  #future::plan("multicore")
  future::plan(sequential)
  # This little thingie took me 3 to 4 hours to find out
  # task$data() uses many cores - up till 10 - cause data.table does some kind of parallelisation!
  # can be turn off like this:
  library(data.table)
  #setDTthreads(threads = 1)
  
  ## Task
  wiki_data = df_wiki[, c('label', 'pair_id', feature_names)]
  wiki_data$label = as.factor(wiki_data$label)
  feature_idxs <- 1:length(feature_names) # mlr3 removes assigned columns from data.table (dt) argument used in pipeop's
  wiki_task = as_task_classif(wiki_data, target = "label", id=paste0("row",row))
  # https://mlr3gallery.mlr-org.com/posts/2020-03-30-stratification-blocking/
  # this makes sure that pairs are together in either train or test groups
  wiki_task$set_col_roles("pair_id", roles = "group")
  
  # train/test split
  set.seed(313)
  
  # column mapping
  # original index -> mlr3 index (sorted by feature name)
  # eg 2 -> 2, 3 (data index) -> 11 (mlr3 index)
  mapping = order(feature_names)

  library("mlr3")
  library(mlr3verse)
  library(mlr3pipelines)
  library(mlr3filters)
  #remotes::install_github("mlr-org/mlr3pipelines@pipeop_filterrows")
  source("PipeOpFilterRows.R") # copied from github branch ... integrated here.
  source("wundt_tuning_functions.R")
  source('pipeop_explode_helpers.R')
  source("PipeOpExplode.R")
  source("PipeOpCaretFindCorrelation.R")

  preprocessing_graph =
    po("select", id="drop_na", param_vals = list(selector = selector_drop_na(drop_na_level) ))  %>>% 
    po("filterrows", param_vals = list(filter_formula = ~ !apply(is.na(.SD), 1, any))) %>>%
    po("classbalancing", id = "undersample", adjust = "major",
       reference = "minor", shuffle = FALSE, ratio = 1 / 1) %>>%
    po("scale")

  if(depth>1) {
    combis = as.data.frame(read_parquet(combi_filename))
    # combis has the feature_idxs of the original data
    # map to mlr3 indexes
    combis$combi.value = lapply(combis$combi, function(v) mapping[v+1])
    # the python combi idxs are 0-bound, R's are 1-bound. Hence the +1 correction.

    if(add_original) {
      for(feature_idx in feature_idxs) {
        combis = combis %>% add_row(combi.value = list(as.integer(feature_idx)), .before=0 )
      }
      combis = combis %>% distinct(combi.value, .keep_all=T)
    }
    combis = combis %>% head(max_features)
   # to save memory, already sort-and-chop the list of combis upon loading

    feature_graph = 
      po("explode", id="explode", param_vals = list(combis = combis )) %>>%
      po("findcorr", id="findcorr", param_vals = list( cutoff=findcorr_cutoff ))

    classifier_graph =
      preprocessing_graph %>>%
      feature_graph %>>%
      po("learner", learner = lrn("classif.cv_glmnet"), param_vals = list(alpha=alpha, lambda.min.ratio=lambda.min.ratio, parallel = FALSE) )

  } else { # depth == 1
    feature_graph = 
      po("findcorr", id="findcorr", param_vals = list( cutoff=findcorr_cutoff ))

    classifier_graph =
      preprocessing_graph %>>%
      #feature_graph %>>% # commented this out on 24-09-2023 -> let lasso handle the feature selection at depth=1
      po("learner", learner = lrn("classif.cv_glmnet"), param_vals = list(alpha=alpha, lambda.min.ratio=lambda.min.ratio, parallel = FALSE) )
  }

      #
  cv5 = rsmp("cv", folds = 5)
  train_test_split = rsmp("holdout", ratio=4/5)
  learner = as_learner(classifier_graph)
  set.seed(seed_value)
  cat(format(Sys.time(), "%H:%M:%S"), " Starting resampling\n")
  rr = resample(wiki_task, learner, train_test_split, store_backends=F)
  cat(format(Sys.time(), "%H:%M:%S"), " Done resampling\n")
  show(rr)
  #extra_columns <- c("acc", "prec", "rec", "f1", "ce", "fn", "fp", "tn", "tp")
  # classif metrics
  classif.metrics = rr$aggregate(c(msr("classif.acc"), msr("classif.ce"), msr("classif.precision"), msr("classif.recall"), msr("classif.fbeta"), msr("classif.fn"), msr("classif.fp"), msr("classif.tn"), msr("classif.tp")))
  #cv5$instantiate(train_scaled_task)
  #cv5$instance
  
  cat(format(Sys.time(), "%H:%M:%S"), " Starting learning\n")
  learner$train(wiki_task)
  cat(format(Sys.time(), "%H:%M:%S"), " Stopping learning\n")
  
  source('target_load_theguardian.R')
  
  # show number of dropped features
  dropped_features = length(learner$model$findcorr$drop_idxs)
  # shows combi features
  all_features = nrow(learner$model$explode$sorted_combis)
  # remaining features
  used_features = length(learner$model$classif.cv_glmnet$train_task$feature_names)
  
  # Get all possible combinations between feature idxs
  #all_combis = all_combinations(feature_idxs, depth)
  #max_features = sum(sapply(all_combis, length))
  
  # Rank combinations based on f-test
  #sorted_combis <- f_test_dt(ppg$explode.output$data()[, 'label':=NULL], ppg$explode.output$data()$label, all_combis) #%>%
  
  learner$predict_type = "prob"
  guardian_pred <- learner$predict_newdata(df_guardian[,feature_names]) # seems to retain order!
  gprob = guardian_pred$prob[,2]
  glogit = log(gprob / (1-gprob)) # https://en.wikipedia.org/wiki/Logit
  gpred <- data.frame(logit=glogit, prob=gprob, stimulus=df_guardian$id )
  
  # Model accuracy
  cat(paste0("Target evaluation on ~",depth,"_",alpha,"\n"))
  merged = merge(guardian_stimulus, gpred, by="stimulus")
  # 3 methods: r_prob, r_logit, and rho
  # 3 variables: noco2, comp, pf (=combined)
  cor_res = list()
  for(var in c('noco2', 'comp', 'pf')) {
    rho = cor(merged[,var], merged$logit, method="spearman")
    r_logit =  cor(merged[,var], merged$logit)
    r_prob = cor(merged[,var], merged$prob)
    cor_res[[paste0('guardian.rho_',var)]] = rho
    cor_res[[paste0('guardian.rlogit_',var)]] = r_logit
    cor_res[[paste0('guardian.rprob_',var)]] = r_prob
  }
  p = merged$prob
  merged$curve = p*(1-p)
  r_inte = cor(merged$inte, merged$curve)
  cat('r (inte^2 merged):', r_inte, '\n')
  r2_inte = summary(lm(inte ~ prob + curve, merged))$r.squared
  cat('r^2 (inte^2 merged):', r2_inte, '\n')

  res = as.list(c(outer_grid[row,], classif.metrics, cor_res))
  res$features.dropped = dropped_features
  res$features.all = all_features
  res$features.used = used_features
  res$guardian.rinte = r_inte
  res$guardian.r2inte = r2_inte
  res$ID = pk_value
  df = data.frame( res )
  df$data_file = as.character(df$data_file)
  df$add_original = as.integer(df$add_original)
  
  # Insert the dataframe into the MySQL table
  drv = dbDriver("MySQL")
  con = dbConnect(drv, user="XXXXX", dbname="com1_ensiwiki-2020_agerank", host=host, port=port, password="XXXXXXXXXX")
  # Create the INSERT statement with column names
  column_names <- paste0("(`", paste(colnames(df), collapse = "`,`"), "`)")
  values <- paste0("('", paste(df, collapse = "','"), "')")
  quoted_column_names = unlist(lapply(names(df), function(x) paste0("`",x,"`")))
  update_statement = paste0("", paste(quoted_column_names, df, sep = " = '", collapse = "', "), "'")
  query <- paste0("INSERT INTO ", table_name, " ", paste(column_names, collapse = ","), " VALUES ", values, " ON DUPLICATE KEY UPDATE ", update_statement)
  dbExecute(con, query)
  # primary key value
  #pk_value <- dbGetQuery(con, "SELECT LAST_INSERT_ID()")$`LAST_INSERT_ID()`
  dbDisconnect(con)
  
    }, error = function(e) {
      print(e)
      return (NULL)
      })
    
  # Graph of p x noco2/inte
  tryCatch({ # fails sometimes
    source('plot_loess.R')
    png(paste0("glmnet_cv/glmnet_cv_",pk_value,"-loess_prob_s75.png"))
    plot_loess('inte', 'prob', merged, span=.75) #default
    dev.off()
    #png(paste0(filename,"_loess_prob_s100.png"))
    #plot_loess('inte.x', 'prob', merged, span=1.0)
    #dev.off()
  }, error = function(e) {print(e)})
  #save(result, file=paste0(dir,"/",combination,"_n-",n,"_alpha-",alpha,"_hco-",hco_threshold,".RData"), version=2)
  #

    # Save model
    if(max_features == max(outer_grid[outer_grid$data==data_file & outer_grid$depth==depth, 'max_features'])) {
      # save models that use all features
      saveRDS(learner, file=gsub(".txt", ".RDS",log_filename))
    }

  cat(format(Sys.time(), "%H:%M:%S"), " Done with row",row," and index",pk_value,"\n")
  return(df)
  }, file = log_filename) # capture output
}  
