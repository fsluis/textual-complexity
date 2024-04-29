package ix.data.com.features

import ix.data.structure.DataSet
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.math.exp

object KenLMFeatures extends FeaturesApp {
  override def appName(dataSet: DataSet) = s"KenLM features (${dataSet.name})"

  override def tableName(dataSet: DataSet) = "features_kenlm" + dataSet.size.map("_" + _).getOrElse("")

  override def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet) = {
    // @transient val dataSet = ix.data.structure.DataSetService("ensiwiki_2020-agerank-1k")
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)
    val content = contentRDDs.last._2

    // Imports
    import org.apache.spark.storage.StorageLevel
    import ix.complexity.lm.KenLM
    import ix.complexity.lm.CommonCrawlPreprocessing

    if (false) {
      //val test = "The small red car turned very quickly around the corner. Joe Smith is from Seattle."
      //val test = "This is a test sentence."
      //val test = " when the sun shines it doesnt rain "
      val test = " when the sun shines it doesn't rain "
      val sentence = CommonCrawlPreprocessing.singleLine(test)
      val scores = KenLM.scoreSentences(sentence, "commoncrawl-en")
      scores.map(_.d)
      scores.map(_.scores.map(_.getLogProbability()))
    }

    val onlyArticles = contentRDDs -- Set("paragraph", "section")
    val dfs = for ((aggName, aggRdd) <- onlyArticles) yield { // <- contentRDDs

      val plainText = aggRdd.vMap(_.plainText)
      val listOfSentences = plainText.vMap(CommonCrawlPreprocessing.text)
      // analysing both text and sentences together, to benefit from caching
      val scores = listOfSentences.
        vMap(sentences => KenLM.scoreSentences(sentences, "commoncrawl-en")).
        persist(StorageLevel.DISK_ONLY)
      val sentenceScores = scores.vFlatMap(_._2.iterator).valueFilter(!_.is_nan_inf())

      // Store results
      import ix.extkey.Features
      @transient val features = new Features()

      // sentence logscale
      features += "cc_prob" -> sentenceScores.vMap(_.d)

      // Add keys
      features.keys ++= keyRDDs.map(_.swap)
      features.keys ++= contentRDDs.map(_.swap)

      // DataFrame
      val (rows, schema, columns, grouped) = features.dataFrame(aggRdd)
      sql.createDataFrame(rows, schema)
    }
    val df = dfs.reduce((df1: DataFrame, df2: DataFrame) => df1.unionAll(df2))
    df
  }
}
