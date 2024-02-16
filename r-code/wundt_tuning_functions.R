library(checkmate)
library(stringr)

# From https://rdrr.io/github/mlr-org/mlr3pipelines/src/R/Selector.R#sym-selector_all
# and https://stackoverflow.com/questions/45325863/how-to-access-hidden-functions-that-are-very-well-hidden
selector_sample = function(n) { 
  mlr3pipelines:::make_selector(function(task) {
    #print(n)
    sample(task$feature_names, n)
  }, paste0("selector_sample(",n,")"), n) 
}

selector_drop_na = function(na) mlr3pipelines:::make_selector(function(task) {
  data_all = task$data()
  na_cols = colSums( is.na(data_all)) >= (nrow(data_all)*na )
  drop_names = names(na_cols[na_cols]) 
  feature_names = task$feature_names[!task$feature_names %in% drop_names] # some features existing in the data are already removed
  if(length(drop_names)>0) {
    cat("Removing", length(drop_names), "columns, from",length(task$feature_names),"to",length(feature_names),"total features, with more than", na*100, "% nan's:", drop_names,"\n") 
  }
  feature_names
}, paste0("selector_drop_na"), na)
