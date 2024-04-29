package ix.data.com.features

import ix.complexity.stanford.MatrixFeatures.matrixToWindows
import ix.complexity.stanford.StanfordCoReferences.corefMatrix
import ix.data.structure.{Content, DataSet}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.ExtKeyFunctions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import no.uib.cipr.matrix.{Matrices, Matrix}
import scala.collection.JavaConversions._
import scala.collection.immutable

object CorefCoherenceFeatures extends FeaturesApp {

  override def appName(dataSet: DataSet) = s"Coref features (${dataSet.name})"

  override def tableName(dataSet: DataSet) = "features_coref" + dataSet.size.map("_" + _).getOrElse("")

  override def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame = {
    // @transient val dataSet = ix.data.structure.DataSetService("ensiwiki_2020-agerank-10")
    import org.apache.spark.rdd.ExtKeyFunctions._
    import org.apache.spark.rdd.ExtKeyTracker
    @transient implicit val tracker = new ExtKeyTracker
    val (keyRDDs, contentRDDs) = dataSet.load(sc)

    import ix.extkey.Features
    import scala.collection.JavaConversions._
    import ix.complexity.stanford.Annotator
    import org.apache.spark.storage.StorageLevel
    if (false) {
      val sentences = List.fill(100)(List("Pete is a man.", "His name is Pete.", "His car is a Toyota.", "He loves driving his car.", "His car is called Petes.")).flatten
      val text = sentences.slice(0, 22).mkString(" ")
      val doc = Annotator(text, "coref_statistical")
      val matrix = corefMatrix(doc, 10)
      new no.uib.cipr.matrix.DenseMatrix(matrix)
      //matrix.map(_.numRows())
    }

    import ix.complexity.lucene3.tokenizer.{Tokenizer => LuceneTokenizer}

    //val aggRdd = contentRDDs("article")
    val onlyArticles = contentRDDs -- Set("paragraph", "section")
    val dfs = for ((aggName, aggRdd) <- onlyArticles) yield {

      val windowSize = 6
      val coref = aggRdd.vMap(con => Annotator(con.plainText, "coref_statistical"))
      val matrices = coref.vMap(corefMatrix(_, windowSize)).persist(StorageLevel.MEMORY_AND_DISK)
      val windows = matrices.kvFlatMap { case (k, v) => matrixToWindows(v, windowSize).iterator.map((k, _)) }

      @transient val features = new Features()

      val maxSteps = 4
      val locals = matrices.vMap(CorefCoherenceFeatures.flatWindows(_, fill = true, maxSteps))
      for (i <- Range.inclusive(0, maxSteps - 1))
        features += s"coref_${i + 1}" -> locals.vMap(_(i)).valueFilter(_.isDefined).vMap(_.get) //.valueFilter(!_.isNaN).valueFilter(!_.isInfinity)

      // add word count (to check for correlation/influence of length)
      // Lucene tokens
      features += "lucene_word_count" -> aggRdd.vMap(con => LuceneTokenizer(con.plainText, "english-lower").size.toDouble)

      // Add keys
      features.keys ++= keyRDDs.map(_.swap)
      features.keys ++= contentRDDs.map(_.swap)

      // DataFrame
      val (rows, schema, columns, grouped) = features.dataFrame(aggRdd, List("mean"))
      sql.createDataFrame(rows, schema)
    }
    val df = dfs.reduce((df1: DataFrame, df2: DataFrame) => df1.unionAll(df2))
    df
  }

  /**
   * Average sum of similarity over n steps in a matrix.
   * Per row, this calculates the sum -1, -2, ... -n steps from the center diagonal
   * The row values at step n are averaged
   * The result is the average similarity at step n, summing 1..n (accumulative)
   * See notes on remarkable, Wundt paper notes p. 50
   *
   * @param matrix
   * @param fill     whether to fill in missing values when the matrix is too small
   * @param maxSteps window size
   * @return indexed sequence, with step n=1 @ index 0
   */
  def flatWindows(matrix: Matrix, fill: Boolean = true, maxSteps:Int = 5): immutable.IndexedSeq[Option[Double]] = {
    for (i <- 1 to maxSteps) yield {
      val startRow = 1
      if (i > matrix.numRows() - 1 && !fill)
        None
      else {
        val rowRes = for (row <- startRow until matrix.numRows()) yield {
          val columns = math.max(row - i, 0) until row
          //println(s"i: $i, row: $row, cols: $columns")
          val values = columns.map(matrix.get(row, _))
          values.sum
        }
        if (rowRes.size > 0)
          Some(rowRes.sum / rowRes.size)
        else
          None
      }
    }
  }

}
