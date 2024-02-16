
PipeOpExplode = R6Class("PipeOpExplode",
  inherit = PipeOpTaskPreproc,
  public = list(
    initialize = function(id = "explode", param_vals = list()) {
      ps = ParamSet$new(params = list(
        #ParamInt$new("max_features", default = -1, tags = c("train", "explode")),
        #ParamInt$new("depth", tags = c("train", "required", "explode")),
        #ParamUty$new("feature_idxs", tags = c("train", "required", "explode"), custom_check = function(x) is.vector(x) ),
        ParamUty$new("combis", tags = c("train", "explode"), custom_check = function(x) is.data.frame(x) )
        #ParamLgl$new("robust", tags = c("train", "required"))
      ))
      #ps$values = list(robust = FALSE, depth = 2)
      super$initialize(id = id, param_set = ps, param_vals = param_vals, feature_types = c("numeric", "integer"))
    }
  ),
  private = list(

    .train_dt = function(dt, levels, target) {
      pv = self$param_set$get_values(tags = "train")
      
      cat(paste(attributes(pv),collapse=","),"\n")
      cat("combis" %in% attributes(pv),"\n")
      # Rank combinations based on f-test
      #if("combis" %in% attributes(pv)) {
        cat("Using pre-set combis\n")
        sorted_combis = pv$combis 
      #} else {
      #  cat("Calculating combis\n")
        # Get all possible combinations between feature idxs
      #  feature_idxs <- 1:ncol(dt) # mlr3 removes assigned columns from data.table (dt) argument used in pipeop's
      #  all_combis = all_combinations(feature_idxs, pv$depth)
        
      #  max_features = sum(sapply(all_combis, length))
      #  if("max_features" %in% attributes(pv))
      #    if(pv$max_features>0)
      #      max_features = pv$max_features
        
      #  combis <- f_test_dt(dt, target, all_combis)
      #  sorted_combis <- combis %>%
      #    arrange(desc(f)) %>%
      #    head(max_features) 
      #}

      # Expand X
      cat(format(Sys.time(), "%H:%M:%S"), " Exploding dt\n")
      dt = explode_dt(dt, sorted_combis)
      self$state = list(sorted_combis = sorted_combis)
      cat(format(Sys.time(), "%H:%M:%S"), " DT rows",nrow(dt),"and ncol",ncol(dt),"\n")
      dt
    },

    .predict_dt = function(dt, levels) {
      sorted_combis = self$state$sorted_combis
      dt = explode_dt(dt, sorted_combis)
      dt
    }
  )
)

mlr_pipeops$add("explode", PipeOpExplode)



