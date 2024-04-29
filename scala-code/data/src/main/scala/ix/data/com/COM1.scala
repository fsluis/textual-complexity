package ix.data.com

import java.util.Properties
import com.google.common.base.Splitter
import de.tudarmstadt.ukp.wikipedia.parser.{ParsedPage, Section}
import ix.common.data.wikipedia.WikiTables
import ix.util.net2.jdbc.{Flushable, MySQLDatabase}
import org.apache.commons.logging.{Log, LogFactory}
import ix.util.net2.spark.SparkDeamon
import ix.common.data.wikipedia.BuildTables.PageRecord
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.rdd.{ExtKey, ExtKeyTracker, RDD}
import org.apache.spark.rdd.ExtKeyFunctions._
import org.apache.hadoop.io.LongWritable
import ix.data.wikipedia.WikipediaParser.{paragraphs, parse, sections}
import ix.extkey.Features
import ix.util.net2.Servers

import scala.collection.immutable

object COM1 extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)
  lazy val database = ix.util.net2.jdbc.MySQLDatabase("kommlabir01fl", "complexity_20110803_nostub").get

  def map2Properties(map: Map[String,String]):java.util.Properties = {
    (new java.util.Properties /: map) {case (props, (k,v)) => props.put(k,v); props}
  }

  lazy val sqlOptions = map2Properties(Map(
    "user" -> "wundt",
    "password" -> "PaSSWoRD",
    "driver" -> "com.mysql.cj.jdbc.Driver",
    "autoReconnect" -> "true",
    "serverTimezone" -> "UTC" //needed for MySQL v. 8, see https://stackoverflow.com/questions/26515700/mysql-jdbc-driver-5-1-33-time-zone-issue
  ))
  // utf-8 encoding of text in java/mysql https://stackoverflow.com/questions/33669612/how-to-properly-set-utf8-encoding-with-jdbc-and-mysql
  lazy val sqlUrl = "jdbc:mysql://localhost/complexity_20110803_nostub_20k?autoReconnect=true&amp;useUnicode=true&amp;characterEncoding=UTF-8"
  // kommlabir01fl complexity_20110803_nostub_20k
  // localhost complexity_20110803_nostub_20k

  def sqlConfig(dataSetName: String): (String, Properties) = {
    //val hostname = Servers.local.map(_.hostname).getOrElse("localhost")
    val hostname = "kommlabir01fl"
    val PhdRE = "phd-(.*)".r
    val EnSiWikiRE = "ensiwiki_(\\d{4})-([a-z/_]+)-([a-z0-9]+)".r
    //val EnSiWikiRE(year, version, size) = "ensiwiki_2020-agerank_nostub-1k"
    dataSetName match {
      case "fokstudy" => (s"jdbc:mysql://$hostname/com1_fokstudy", sqlOptions)
      case "theguardian" => (s"jdbc:mysql://$hostname/com1_theguardian", sqlOptions)
      case "theguardian-all" => (s"jdbc:mysql://$hostname/com1_theguardian_all", sqlOptions)
      case PhdRE(size) =>
        (s"jdbc:mysql://$hostname/complexity_20110803_nostub_20k", sqlOptions)
      case EnSiWikiRE(year, version, size) =>
        (s"jdbc:mysql://$hostname/com1_ensiwiki-${year}_${version.replaceAll("_", "-")}", sqlOptions)
      case _ =>
        ("", new Properties)
    }

  }

}