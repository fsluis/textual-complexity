package ix.data.com.features

import ix.data.structure.DataSet
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

object WordFeatures extends FeaturesApp {

  override def appName(dataSet: DataSet) = s"Word features (${dataSet.name})"
  override def tableName(dataSet: DataSet) = "features_word" + dataSet.size.map("_"+_).getOrElse("")

  def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame = {
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)

    import ix.complexity.hyphen.Hyphenator
    import ix.complexity.lucene3.tokenizer.{Tokenizer => LuceneTokenizer}
    import ix.extkey.Features

    import scala.collection.JavaConversions._
    @transient val features = new Features()
    val content = contentRDDs.last._2

    // Lucene tokens
    val lucTextTokens = content.vMap( c => LuceneTokenizer.tokenize(c.plainText, "english-lower", c.charIndex) )
    val lucTokens = lucTextTokens.kvFlatMap(_._2)
    features += "lucene_characters_per_word" -> lucTokens.vMap(_.length)
    features += "lucene_syllables_per_word" -> lucTokens.vMap( word => Hyphenator.syllables(word ,"english").length )

    // Add keys
    features.keys ++= keyRDDs.map(_.swap)
    features.keys ++= contentRDDs.map(_.swap)

    // Transform to DataFrame
    val dfs = contentRDDs.values.map(aggRDD => features.dataFrame(aggRDD, List("mean", "sum"))).map{ case (r,s,c,q) => sql.createDataFrame(r,s) }
    val df = dfs.reduce((df1: DataFrame, df2: DataFrame) => df1.unionAll(df2))
    df
  }

}
