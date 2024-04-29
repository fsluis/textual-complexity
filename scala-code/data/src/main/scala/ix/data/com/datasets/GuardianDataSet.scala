package ix.data.com.datasets

import ix.data.structure.{Content, DataSet, Text}
import org.apache.commons.configuration.Configuration
import org.apache.spark.SparkContext
import org.apache.spark.rdd.{ExtKey, ExtKeyTracker, RDD}
import ix.common.data.guardian.GuardianConfig.GuardianRecord
import org.apache.hadoop.io.LongWritable
import org.apache.spark.rdd.ExtKeyFunctions._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.immutable.ListMap

class GuardianDataSet extends DataSet {
  var filename: String = System.getProperty("user.home") + "resources/data/guardian_selection.hsf"
  var size: Option[String] = None
  var name: String = ""

  override def init(name: String, conf: Configuration): Unit = {
    filename = conf.getString("filename", filename)
    this.name = name
  }

  override def load(sc: SparkContext)(implicit tracker: ExtKeyTracker): (ListMap[String, RDD[_]], ListMap[String, RDD[(ExtKey[Long], Content)]]) = {
    // Get content
    val data = sc.sequenceFile(filename, classOf[LongWritable], classOf[GuardianRecord], 100)
    val pages = ext(data.map{case (k,v) => (k.get, v)})
    val content: RDD[(ExtKey[Long], Content)] = pages.kvMap{case (id, r) => (r.getLong("id").toLong, Text(r.getString("content")) )}

    //: Map[String, RDD[(Long, Content)]]
    val contentRDDs = ListMap("article" -> content)
    val keyRDDs: ListMap[String, RDD[_]] = ListMap("page_id" -> pages, "article_id" -> content)
    (keyRDDs, contentRDDs)
  }
}
