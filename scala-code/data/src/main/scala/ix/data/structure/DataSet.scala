package ix.data.structure

import org.apache.commons.configuration.Configuration
import org.apache.spark.SparkContext
import org.apache.spark.rdd.{ExtKey, ExtKeyTracker, RDD}

import scala.collection.immutable.ListMap
import scala.collection.mutable

abstract class DataSet {
  def init(name: String, conf: Configuration): Unit

  def load(sc: SparkContext)(implicit tracker: ExtKeyTracker): (ListMap[String, RDD[_]], ListMap[String, RDD[(ExtKey[Long], Content)]])

  def name: String
  def size: Option[String]
}
