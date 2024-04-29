package ix.data.com.features

import ix.data.structure.{Content, DataSet}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.storage.StorageLevel

object WordNetFeatures extends FeaturesApp {
  override def appName(dataSet: DataSet) = s"WordNet features (${dataSet.name})"

  override def tableName(dataSet: DataSet) = "features_wordnet" + dataSet.size.map("_" + _).getOrElse("")

  override def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame = {
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)
    val content = contentRDDs.last._2

    // Imports
    import scala.collection.JavaConversions._
    import ix.complexity.lucene3.tokenizer.{Tokenizer => LuceneTokenizer}
    import ix.extkey.Features
    import ix.complexity.wordnet.WordNetService
    import ix.complexity.wordnet.model.{Node, StOngeNode}

    // Pre-processing
    val words = content.kvFlatMap { case (index, con) => LuceneTokenizer.tokenize(con.plainText, "english-lower", index) }

    @transient val features = new Features()
    val maxDepth = 5
    val nodes = words.vMap(WordNetService.findNode(_, WordNetService.ALL_POS, classOf[StOngeNode], "wordnet-3.1"))
    val filtered = nodes.valueFilter(_.getSynsets.nonEmpty) //removes non-found words
    val spread = filtered.vMap(WordNetService.computerSpread(_, maxDepth, "wordnet-3.1")).
      persist(StorageLevel.MEMORY_AND_DISK)

    for (depth <- 0 until maxDepth) {
      features += s"wordnet_${depth + 1}" -> spread.vMap(arr => arr(depth)).valueFilter(_ != null).vMap(_.toDouble).vMap(math.log)
    }

    // Add keys
    features.keys ++= keyRDDs.map(_.swap)
    features.keys ++= contentRDDs.map(_.swap)

    // Transform to DataFrame
    val dfs = contentRDDs.values.map(aggRDD => features.dataFrame(aggRDD)).map { case (r, s, c, q) => sql.createDataFrame(r, s) }
    val df = dfs.reduce((df1: DataFrame, df2: DataFrame) => df1.unionAll(df2))
    df
  }
}
