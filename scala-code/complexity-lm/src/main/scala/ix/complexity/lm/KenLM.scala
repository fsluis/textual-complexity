package ix.complexity.lm

import com.github.jbaiter.kenlm.jni.LoadMethod
import com.github.jbaiter.kenlm.{Config, FullScoreReturn, Model}
import ix.common.util.TimeLogger
import org.apache.spark.SparkFiles
import ix.util.services2.{ModelService, Models, ServiceException}
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}

//import collection.JavaConverters
import collection.JavaConversions.collectionAsScalaIterable

object KenLM extends ModelService[Model] {
  val bos = config.getBoolean("use_bos", true)
  val eos = config.getBoolean("use_eos", true)
  val log: Log = LogFactory.getLog(this.getClass)
  val tlog: TimeLogger = new TimeLogger(log, "kenlm parsing", 1000)

  override def load(name: String, confOpt: Option[Configuration]): Model = {
    System.setProperty("com.github.jbaiter.kenlm.lib.name", "libkenlm-jni.so")
    System.setProperty("com.github.jbaiter.kenlm.lib.path", System.getProperty("user.home")+"/farm/src/kenlm-java")

    // Need to pre-load KenLM like this to fix UnsattisfiedLinkErrors!
    com.github.jbaiter.kenlm.util.KenLMLoader.load()
    val conf = new Config()
    conf.setLoadMethod(LoadMethod.LAZY)
    new Model(confOpt.get.getString("file", "no file set"), conf)
  }

  def scores(sentences: Array[String], modelName: String = Models.DEFAULT, oov: Boolean = true) = {
    val model = get(modelName)
    val scores = collectionAsScalaIterable(model.fullScores(sentences, bos, eos))
    if(oov)
      scores.map(_.getLogProbability)
    else
      scores.filter(!_.isOov).map(_.getLogProbability)
  }

  def scoreSentences(sentences: Seq[String], modelName: String = Models.DEFAULT): Seq[LMScores] = {
    for(sen <- sentences) yield
      scoreText(sen, modelName)
  }

  def scoreText(text: String, modelName: String = Models.DEFAULT): LMScores = {
    val model = get(modelName)
    //val scores = model.fullScores(sentence.toArray)
    //for(score <- scores) {
    //  score
    //}
    //scores
    // don't give array of strings / sentences, this doesn't return word-level log probabilities
    val t = tlog.start()
    val scores = collectionAsScalaIterable(model.fullScores(text, bos, eos))
    tlog.stop(t)
    //for(sentenceScores <- scores)
    LMScores(scores)
  }

}

case class LMScores(scores: Iterable[FullScoreReturn]) {
  val d: Double = scoresToResults( scores.map(_.getLogProbability.toDouble) )

  def is_nan_inf(): Boolean = {
    d.isNaN || d.isInfinity
  }

  def scoresToResults(logs: Iterable[Double]): Double = {
    // Geometric mean in logscale of p
    val geo_log_p = logs.sum / logs.size

    geo_log_p
  }
}

/*
GET HUGE LM
From https://statmt.org/ngrams/
wget http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/nl/lm/nl.2012-2013.trie.xz -O - | unxz > wget.nl.trie

KENLM BUILD LOG
git clone https://github.com/jbaiter/kenlm-java.git
# note: added Serializable interface to FullScoreReturn
kommlabir01fl:~/farm/src/kenlm-java$ export CXXFLAGS="-I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux -I~/farm/src/kenlm -L~/farm/src/kenlm/build/lib"
kommlabir01fl:~/farm/src/kenlm-java$ ./build.sh
export LD_LIBRARY_PATH="~/farm/src/kenlm-java/"

import ix.complexity.lm.KenLM
ix.complexity.lm.KenLM.scores(Array("dit is een test zin"), "commoncrawl-nl")
 */