package ix.data.com.features

import ix.data.structure.{DataSet, DataSetService}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import ix.complexity.features.common.SlidingWindowEntropy
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics
import org.apache.spark.rdd.{ExtKey, ExtKeyTracker, RDD}
import org.apache.spark.rdd.ExtKeyFunctions._

import scala.collection.JavaConversions._
import scala.reflect.ClassTag

object SWEFeatures extends FeaturesApp {

  override def appName(dataSet: DataSet) = s"SWE features (${dataSet.name})"
  override def tableName(dataSet: DataSet) = "features_swe" + dataSet.size.map("_"+_).getOrElse("")

  def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame = {
    // @transient val dataSet = ix.data.structure.DataSetService("ensiwiki_2020-agerank-50k")
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)

    import ix.complexity.lucene3.esa.{CosineKernel, EsaService}
    import ix.complexity.lucene3.tokenizer.{Tokenizer => LuceneTokenizer}
    import ix.extkey.Features
    import ix.complexity.features.word.EsaFeatures
    import org.apache.spark.storage.StorageLevel
    val content = contentRDDs.last._2
    val topicsList = content.
      kvvMap { case (index, text) => EsaService.topics(text.plainText, "english", index).filter(_._2.isDefined).map(kv => kv._1 -> kv._2.get).toList }.
      valueFilter(_.nonEmpty).
      persist(StorageLevel.MEMORY_AND_DISK) // DISK_ONLY
    // Tip: Don't forget to remove empty lists (ie content where no topic could be found)
    // otherwise this can lead to NaN later on when comparing topics (semantic relatedness)

    val onlyArticles = contentRDDs -- Set("paragraph", "section")
    val dfs = for ( (aggName, aggRdd) <- onlyArticles) yield { //  onlyArticles contentRDDs
      // ESA SWE feature
      @transient val features = new Features()
      val topics = topicsList.dropKey(aggRdd).extReduceByKey(_++_).vMap( _.sortBy(_._1).map(_._2) ) //.persist(StorageLevel.DISK_ONLY)
      val topicWindows = Seq(10, 15, 20, 25, 30)
      for(windowSize <- topicWindows ) {
        val windows = topics.kvFlatMap( kv => EsaFeatures.windowedSums(kv._2, windowSize) )
        val swe = windows.vMap( centroid => EsaFeatures.entropy(centroid) )
        features += s"esa_swe_w$windowSize" -> swe
      }

      // Word SWE features
      // snowball, but no stopword removal
      val tokenWindows = Seq(45, 50, 55, 60, 65)
      val snowball = aggRdd.vMap( c => LuceneTokenizer(c.plainText, "english-lower-stem").map(_._2).toList ).persist(StorageLevel.MEMORY_AND_DISK) //MEMORY_AND_DISK
      for(windowSize <- tokenWindows ) {
        val tokenSweRdds = slidingWindowEntropies(snowball, 5, windowSize)
        tokenSweRdds.zipWithIndex.foreach{case (rdd, index) => features += s"snowball_swe_w${windowSize}_n${index+1}" -> rdd }
      }

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

  def slidingWindowEntropies[K, V](tokens: RDD[(ExtKey[K], List[V])], n: Int = 5, size: Int = 25)
                                  (implicit evk: ClassTag[K], evv: ClassTag[V], tracker: ExtKeyTracker): IndexedSeq[RDD[(ExtKey[Int], Double)]] = {
    val swe = tokens.vMap((values: List[V]) => SlidingWindowEntropy.calculateEntropy(values, n, size))
    for (i <- 0 until n) yield swe.vFlatMap((tuple: (K, Array[DescriptiveStatistics])) => tuple._2(i).getValues.toIterator)
  }
}
