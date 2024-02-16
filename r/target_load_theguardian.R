library('DBI')
library('RMySQL')

# Get interest data
guardian_raw=read.table("data/guardian/glasgow_production_20190824.csv", sep=",", header=TRUE)

# Inverse scales
guardian_raw$novelty_comprehension_scale_59 = -1*guardian_raw$novelty_comprehension_scale_59+8 #:left => 'easy to read', :right => 'difficult to read'
guardian_raw$novelty_comprehension_scale_60 = -1*guardian_raw$novelty_comprehension_scale_60+8 #:left => 'easy to understand', :right => 'hard to understand'
guardian_raw$novelty_comprehension_scale_61 = -1*guardian_raw$novelty_comprehension_scale_61+8 #:left => 'comprehensible', :right => 'incomprehensible'
guardian_raw$novelty_comprehension_scale_62 = -1*guardian_raw$novelty_comprehension_scale_62+8 #:left => 'coherent', :right => 'incoherent'
guardian_raw$novelty_comprehension_scale_63 = -1*guardian_raw$novelty_comprehension_scale_63+8 #:left => 'interesting', :right => 'uninteresting'

# Summarize
guardian_all = data.frame(
  session=guardian_raw$session,
  stimulus=guardian_raw$stimulus,
  condition=guardian_raw$condition,
  inte=(guardian_raw$novelty_comprehension_scale_63+guardian_raw$novelty_comprehension_scale_64+guardian_raw$interest_scale_67)/3,
  read=(guardian_raw$novelty_comprehension_scale_57+guardian_raw$novelty_comprehension_scale_59)/2,
  noco2=(guardian_raw$novelty_comprehension_scale_57+guardian_raw$novelty_comprehension_scale_59)/-2+8,
  fama=guardian_raw$novelty_comprehension_scale_58,
  famt=guardian_raw$familiarity,
  fam=rowMeans(guardian_raw[,c("fam_answer_1", "fam_answer_2", "fam_answer_3", "novelty_comprehension_scale_58")], na.rm=TRUE),
  position=guardian_raw$position,
  comp=(guardian_raw$novelty_comprehension_scale_60 + guardian_raw$novelty_comprehension_scale_61 + guardian_raw$novelty_comprehension_scale_62)/3,
  pf = (guardian_raw$novelty_comprehension_scale_57 + guardian_raw$novelty_comprehension_scale_59 + guardian_raw$novelty_comprehension_scale_60 + guardian_raw$novelty_comprehension_scale_61 + guardian_raw$novelty_comprehension_scale_62)/5
)
guardian_all=na.omit(guardian_all)

# Add condition_position column
for(i in unique(guardian_all$session)) {
  for(j in c("low", "high", "misc")) {
    guardian_all[which(guardian_all$session==i & guardian_all$condition==j),'condition_position'] = 1:6
  }}

# Aggregate by session
guardian_session=aggregate(guardian_raw, by=list(guardian_raw[,'session']), FUN=mean, na.rm=TRUE)
guardian_session$ec_d = guardian_session$ec_scale_120+guardian_session$ec_scale_121+guardian_session$ec_scale_122+guardian_session$ec_scale_123+guardian_session$ec_scale_124
guardian_session$ec_s = guardian_session$ec_scale_125+guardian_session$ec_scale_126+guardian_session$ec_scale_127+guardian_session$ec_scale_128+guardian_session$ec_scale_129
guardian_session$ec = guardian_session$ec_d+guardian_session$ec_s

# Merge with main data
guardian_all = merge(guardian_session[c("session", "ec_d", "ec_s", "ec")], guardian_all, by.x='session')

# Aggregate data per stimulus
guardian_stimulus=aggregate(guardian_all, by=list(guardian_all[,'stimulus']), FUN=mean, na.rm=TRUE)  
