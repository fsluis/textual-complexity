
library(dplyr)
library(tidyr)
library(purrr)
library(magrittr)

# all_combinations function
all_combinations <- function(feature_idxs, depth, auto_interactions=FALSE) {
  # depth=2 -> [2], depth=3 -> [2,3]
  # original features
  all_combis <- list()
  all_combis[[1]] <- lapply(feature_idxs, function(i) c(i))
  
  if(auto_interactions) {
    all_combis[[2]] <- lapply(feature_idxs, function(i) c(i,i))
  }
  
  # two- and three-way interactions
  c <- lapply(seq_len(depth-1)+1, function(d) combn(feature_idxs, d, simplify = FALSE))
  all_combis <- c(all_combis, c)
  
  return(all_combis)
}

# calc_combi function
calc_combi <- function(X, combi) {
  if(length(combi)==1) {
    x <- X[, combi]
  } else if(length(combi)==2) {
    x <- X[, combi[1]] * X[, combi[2]]
  } else if(length(combi)==3) {
    x <- X[, combi[1]] * X[, combi[2]] * X[, combi[3]]
  }
  return(x)
}

calc_combi_dt <- function(X, combi) {
  if(length(combi)==1) {
    x <- X[, ..combi]
  } else if(length(combi)==2) {
    x <- X[ , combi[1], with=FALSE ] * X[ , combi[2], with=FALSE ]
  } else if(length(combi)==3) {
    x <- X[ , combi[1], with=FALSE ] * X[ , combi[2], with=FALSE ] * X[ , combi[3], with=FALSE ]
  }
  x[[1]]
}


# f_test function
f_test_dt <- function(X, y, all_combis) {
  out_features <- sum(sapply(all_combis, length))
  fs <- matrix(0, nrow = out_features, ncol = 2)
  cs <- list()
  i <- 1
  
  for(combis in all_combis) {
    for(combi in combis) {
      x <- calc_combi_dt(X, combi)
      res.aov <- aov(x ~ y)
      res.sum = summary(res.aov)
      fs[i,1] <- res.sum[[1]]$`F value`[1] # f value
      fs[i,2] <- res.sum[[1]]$`Pr(>F)`[1] # p value
      cs[[i]] <- unlist(combi)
      i <- i+1
    }
  }
  
  return(data.frame(combi = enframe(cs), f = fs[,1], p = fs[,2]))
}


explode_dt = function(dt, sorted_combis) {
  # Expand X
  dt_expanded <- matrix(0, nrow = nrow(dt), ncol = nrow(sorted_combis))
  for (i in 1:nrow(sorted_combis)) {
    combi = unlist(sorted_combis[i,'combi.value'])
    if(length(combi)==0) {
      cat("found empty combi at row",i) # shouldn't happen anymore, after fixing the diff in 0-bound / 1-bound indexes between python and R
      next
    }
      x <- calc_combi_dt(dt, combi) 
    dt_expanded[,i] = x
  }
  
  # assign column names
  cnames = unlist(lapply(sorted_combis[,'combi.value'], function(x) paste0(x,collapse=":")))
  colnames(dt_expanded) <- cnames
  
  # return values / update state
  dt = as.data.table(dt_expanded)
  setnames(dt, cnames)
  dt
}

