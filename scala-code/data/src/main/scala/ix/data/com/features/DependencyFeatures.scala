package ix.data.com.features

import ix.complexity.stanford.DependencyAnnotator
import ix.data.structure.{Content, DataSet}
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

object DependencyFeatures extends FeaturesApp {

  override def appName(dataSet: DataSet) = s"Dependency features (${dataSet.name})"

  override def tableName(dataSet: DataSet) = "features_locality" + dataSet.size.map("_" + _).getOrElse("")

  override def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame = {
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)
    val content = contentRDDs.last._2

    // Pre-processing
    import edu.stanford.nlp.ling.TaggedWord
    import ix.complexity.stanford.{Dependency, Annotator => StanfordAnnotator}
    import org.apache.spark.storage.StorageLevel

    import scala.collection.JavaConversions._
    import scala.collection.immutable.SortedMap
    // we cannot use the "depparse" annotator here because of serialization issues on SemanticGraph/GrammanticalRelation
    // note: StanDoc itself is also not serializable
    val sentences = content.kvFlatMap { case (index, con) => DependencyAnnotator(con.plainText, index) }.persist(StorageLevel.MEMORY_AND_DISK)

    // Tests
    if (false) {
      //val test = "The small red car turned very quickly around the corner. Joe Smith is from Seattle."
      val test = "Dependency length counted in nouns, verbs, and noun phrases within a dependency."
      //val test = "Dependency length counted in shallow trees and big trees within a dependency."
      //val test = "Dependency length was nicely counted in shallow trees and big trees within a dependency."
      //val test = "He left his grammar book in the study."
      //val test = "The reporter who the senator who John met attacked disliked the editor." //4x 3, 1x 6
      //val text = "John met the senator who attacked the reporter who disliked the editor." //1x 5
      //val test = "We left the shore the week before."
      import ix.complexity.stanford._
      val sens = DependencyAnnotator(test, debug = true)
      sens.head._2.locality(correction = 0, debug = true)
    }

    // Features
    import ix.extkey.Features
    @transient val features = new Features()
    features += "wps" -> sentences.vMap(_.tokens.size())

    val (correction, filter) = (2, 0)
    // correction: correct by -2 for this to exclude dependencies of length 1 (that don't have a head and dep)
    // filter: only include dependencies of 0 or more length (removes negative length values after correction)

    // parse distances
    val distances = sentences.vMap(sen => ix.data.com.features.TokensDistances(sen.tokens.length, sen.locality(correction).map(_.productElement(0).asInstanceOf[Int])))
    val filtered = distances.vMap(dd => dd.filtered(filter))

    // features
    // note that this features is per sentences and will be summarized later over all sentences
    // alternative would be to flatmap the per-sentence features and take an average over all dependencies/sentences
    val logDistances = filtered.vMap(dd => dd.distances.map(d => math.log10(d + 1)))
    // log i+1 because often the dependency lengths are 0, which is not possible (and log(1)=0)
    val logStats = logDistances.vMap((dd: Seq[Double]) => new DescriptiveStatistics(dd.toArray))

    features += s"deploc_log_mean" -> logStats.vMap(_.getMean()).valueFilter(!_.isNaN)

    // Add keys
    features.keys ++= keyRDDs.map(_.swap)
    features.keys ++= contentRDDs.map(_.swap)

    // Transform to DataFrame
    val dfs = contentRDDs.values.map(aggRDD => features.dataFrame(aggRDD)).map { case (r, s, c, q) => sql.createDataFrame(r, s) }
    val df = dfs.reduce((df1: DataFrame, df2: DataFrame) => df1.unionAll(df2))
    df
  }

}

case class TokensDistances(tokenCount: Int, distances: Seq[Int]) {
  def stats = new DescriptiveStatistics(distances.map(_.toDouble).toArray)

  def mdd = if (tokenCount >= 2) stats.getSum / (tokenCount - 1) else Double.NaN

  def filtered(filter: Int) = TokensDistances(tokenCount, distances.filter(d => d >= filter))
}
