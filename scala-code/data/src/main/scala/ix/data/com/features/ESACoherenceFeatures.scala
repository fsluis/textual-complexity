package ix.data.com.features

import java.lang
import gnu.trove.map.hash.TIntDoubleHashMap
import ix.complexity.lucene3.esa.{CosineKernel, Kernel}
import ix.complexity.stanford.MatrixFeatures.matrixToWindows
import ix.complexity.stanford.ReferenceFeatures.matrixToFeatures
import ix.data.structure.DataSet
import ix.data.com.COM1.log
import ix.extkey.Features
import ix.util.net2.Servers
import ix.util.net2.spark.SparkDeamon
import org.apache.spark.SparkContext
import org.apache.spark.rdd.ExtKeyFunctions._
import org.apache.spark.rdd.{ExtKey, ExtKeyTracker, RDD}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.storage.StorageLevel

import java.util.Arrays
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

/**
 * Note: ESA / Trove does not work with Kryo serializer!
 */
object ESACoherenceFeatures extends FeaturesApp {


  override def appName(dataSet: DataSet) = s"ESA coherence features (${dataSet.name})"

  override def tableName(dataSet: DataSet) = "features_esacoh" + dataSet.size.map("_" + _).getOrElse("")

  override def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame = {
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)

    import ix.extkey.Features
    import scala.collection.JavaConversions._
    import ix.complexity.features.word.EsaFeatures
    import ix.complexity.lucene3.esa.EsaService
    import ix.complexity.stanford.Annotator
    import org.apache.spark.storage.StorageLevel
    if (false) {
      val text = "This is a text about words. This is a text about words. This is a text about words. This is a text about words. This is a text about words. This is a text about words. This is a text about words. This is a text about words. This is a text about words. This is a text about words. "
      val doc = Annotator(text, "splitter")
      val sentences = doc.sentences().map(s => s.charOffsets().first -> s.text)
      val topics = sentences.map { case (index: Integer, sen: String) => EsaService.topics(sen, "english", 0).map(_._2).map(_.get) }
      val centroids = topics.map(EsaFeatures.centroid)
      val kernel = new CosineKernel
      kernel.compare(centroids(0), centroids(4))

      for (i <- 0 until centroids.size - 2) yield
        kernel.compare(centroids(i), centroids(i + 2))
      val matrix = kernel.matrixx(centroids, 1)
      val analysis = matrixToFeatures(matrix, 3, false, false, false, false)
    }
    val content = contentRDDs.last._2

    // Split to sentences
    val stanDoc = content.vMap(c => Annotator(c.plainText, "splitter"))
    val stanSentences = stanDoc.kvFlatMap { case (index, doc) => doc.sentences().map(s => index + s.charOffsets().first -> s.text) }

    // Get topics
    val topicOptions = stanSentences.kvvMap { case (index: Long, sen: String) => EsaService.topics(sen, "english", index).toList } // toList is necessary to resolve a serialization error on lucene's StopFilter
    val topics = topicOptions.vMap(_.map(_._2).filter(_.isDefined).map(_.get)).valueFilter(_.nonEmpty)
    // Tip: Don't forget to remove empty lists (ie content where no topic could be found)
    // otherwise this can lead to NaN later on when comparing topics (semantic relatedness)
    // Get centroid of each sentence, listed by doc: [charIndex, sentence-centroid]
    val centroids = topics.vMap(EsaFeatures.centroid).disentangle(stanDoc) //.persist(StorageLevel.MEMORY_AND_DISK) < this one turns huge (250GB)

    // for token count
    //val senLen = sen.vMap( LuceneTokenizer(_, "english-lower").size )
    val tokenCount = stanDoc.vMap(_.tokens().size())

    // val aggRdd = contentRDDs("paragraph")
    val dfs = for ((aggName, aggRdd) <- contentRDDs) yield {
      @transient val features = new Features()
      val topics = centroids.groupByKey(aggRdd).vMap(_.toList.sortBy(_._1).map(_._2)) //.persist(StorageLevel.MEMORY_AND_DISK)
      // I'm pretty sure the output of a groupBy operation gets cached/saved on disk. It's better not to do this explicitly here, as it
      // leads to an overflow of available disk space (500G...).

      val maxSteps = 4
      val kernel = new CosineKernel
      val matrices = topics.vMap(ts => kernel.matrixx(ts, maxSteps)).persist(StorageLevel.MEMORY_AND_DISK)

      val locals = matrices.vMap(CorefCoherenceFeatures.flatWindows(_, fill = true, maxSteps))
      for (i <- Range.inclusive(0, maxSteps - 1))
        features += s"esacoh_${i + 1}" -> locals.vMap(_(i)).valueFilter(_.isDefined).vMap(_.get) //.valueFilter(!_.isNaN).valueFilter(!_.isInfinity)

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




