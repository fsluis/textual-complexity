package ix.data.com.features

import ix.data.structure.DataSet
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.storage.StorageLevel

/**
 * Attempt to extend / redo some of the analysis from PhD period
 * Focus on connectives
 */
object AltLexFeatures extends FeaturesApp {

  override def appName(dataSet: DataSet) = s"AltLex features (${dataSet.name})"

  override def tableName(dataSet: DataSet) = "features_altlex" + dataSet.size.map("_"+_).getOrElse("")

  override def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame = {
    //@transient val dataSet = ix.data.structure.DataSetService("theguardian")
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)
    val content = contentRDDs.last._2

    // Tests
    import ix.complexity.python.AltLexParser
    if(false) {
      val res = AltLexParser.parse("pythontestpipe", "The sole official language in Turkey is Turkish. It belongs to the Turkic language group, which also includes many other languages spoken across Asia, such as Azerbaijani and Tatar. In Turkey there are also minorities who speak languages such as Kurdish, Armenian, Greek or Ladino, to name just a few.")
      AltLexParser.analyze("pythontestpipe", "Frequency of words belonging to causal connectives (e.g., because, therefore, due to) and non-causal connectives (e.g., and, but, after).")
      AltLexParser.analyze("pythontestpipe", "This happened as a result of his behavior.")
      val text = content.keyFilter(_==10879).first._2.plainText
      AltLexParser.parse("pythontestpipe", text)
      content.vMap( con => AltLexParser.parse("pythontestpipe", con.plainText) ).first()
    }

    // Parse altlexes
    val altlexes = content.vMap( con => AltLexParser.parse("pythontestpipe", con.plainText) ).
      valueFilter(_.isDefined).vMap(_.get).persist(StorageLevel.MEMORY_AND_DISK_SER_2)

    // Preparing features
    import ix.extkey.Features
    @transient val features = new Features()
    features += "nr_of_words" -> altlexes.vMap(res => res.nrOfWords)
    features += "causal_word_span" -> altlexes.vMap(res => res.causalWordSpan)
    features += "non_causal_word_span" -> altlexes.vMap(res => res.nonCausalWordSpan)

    // Add keys
    features.keys ++= keyRDDs.map(_.swap)
    features.keys ++= contentRDDs.map(_.swap)

    // Transform to DataFrame
    val dfs = contentRDDs.values.map(aggRDD => features.dataFrame(aggRDD, List("sum"))).map{ case (r,s,c,q) => sql.createDataFrame(r,s) }
    var df = dfs.reduce((df1: DataFrame, df2: DataFrame) => df1.unionAll(df2))

    // Expand dataframe with derived features
    import org.apache.spark.sql.functions.udf
    val ratio = udf( (numerator: Int, denominator: Int) => if(denominator >0) Some(numerator.toDouble / denominator) else None ) // Double.NaN and null both give errors while saving
    // #causal connectives per #words
    df = df.withColumn("causal_word_span_ratio", ratio(df("causal_word_span_sum"), df("nr_of_words_sum")) )
    // #noncausal connectives per #words
    df = df.withColumn("noncausal_word_span_ratio", ratio(df("non_causal_word_span_sum"), df("nr_of_words_sum")) )

    // Return
    df
  }
}
