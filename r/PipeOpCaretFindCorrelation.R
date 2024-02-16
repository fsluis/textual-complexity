
PipeOpCaretFindCorr = R6Class("PipeOpCaretFindCorr",
                        inherit = PipeOpTaskPreproc,
                        public = list(
                          initialize = function(id = "findcorr", param_vals = list()) {
                            ps = ParamSet$new(params = list(
                              ParamDbl$new("cutoff", tags = c("train", "findcorr"))
                            ))
                            ps$values = list(cutoff = .9)
                            super$initialize(id = id, param_set = ps, param_vals = param_vals, feature_types = c("numeric", "integer"))
                          }
                        ),
                        private = list(
                          
                          .train_dt = function(dt, levels, target) {
                            cat(format(Sys.time(), "%H:%M:%S"), " FindCorrelation start\n")
                            pv = self$param_set$values
                            #pv = self$param_set$get_values(tags = "train")
                            cm = invoke(stats::cor,
                                        x = dt) # ,.args = pv
                            cm = abs(cm) # correlation matrix
                            
                            #cm = cor(dt)
                            drop_idxs = caret::findCorrelation(cm, cutoff=pv$cutoff)
                            self$state = list(drop_idxs = drop_idxs)
                            cat(format(Sys.time(), "%H:%M:%S"), " FindCorrelation done\n")
                            dt[,(drop_idxs):=NULL]
                          },
                          
                          .predict_dt = function(dt, levels) {
                            drop_idxs = self$state$drop_idxs
                            dt[,(drop_idxs):=NULL]
                          }
                        )
)

library(caret)
mlr_pipeops$add("findcorr", PipeOpCaretFindCorr)



