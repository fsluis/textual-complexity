package ix.data.com.datasets

import com.databricks.spark.xml._
import ix.data.com.COM1.log
import ix.util.net2.Servers
import ix.util.net2.spark.SparkDeamon
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

/**
 * Adds page_revisions table to JWPL's wiki database
 * Uses enwiki-20200401-stub-meta-history*.xml.gz files as input
 * Make sure to add enough heap space, fx:
 * nohup spark-submit --driver-memory 6G --executor-memory 40G --jars $ADD_JARS --class ix.data.com.datasets.WikiRevisionsTable --properties-file $HOME/farm/etc/kommlab/spark.conf --files $FILES data/target/scala-2.11/data-assembly-0.1-SNAPSHOT.jar &
 */
object WikiRevisionsTable extends App {
  override def main(args: Array[String]): Unit = {
    log.info("Loading Spark/SQL")
    val daemon = SparkDeamon(Servers.localhost).get
    val sc: SparkContext = daemon.context("WikiRevisionsTable")
    val sqlContext = new SQLContext(sc)
    run(sc, sqlContext, "english2020")
    // Closing is needed because of http://stackoverflow.com/questions/28362341/error-utils-uncaught-exception-in-thread-sparklistenerbus
    log.info("Closing Spark")
    daemon.stopSpark()
  }

  def run(sc: SparkContext, sql: SQLContext, wiki: String) = {

    val home = System.getProperty("user.home")
    val (filename, sqlUrl) = wiki match {
      case "simple2020" => (home+"/local/data/wiki/simplewiki20200401/simplewiki-20200401-stub-meta-history*.xml.gz", "jdbc:mysql://kommlabir01fl/wiki_simple_20200401")
      case "english2020" => (home+"/local/data/wiki/enwiki20200401/enwiki-20200401-stub-meta-history*.xml.gz", "jdbc:mysql://kommlabir01fl/wiki_english_20200401")
      case _ => ("","")
    }

    // Load xmls
    val pageDf = sql.read.option("rowTag", "page").xml(filename)

    // Turn nested structure of page:revision into a flat table
    val revisionDf = pageDf.
      selectExpr("id as page_id", "ns as page_namespace", "title as page_title", "explode(revision) as r").
      selectExpr("page_id", "page_namespace", "page_title", "r.id as revision_id", "r.parentid as revision_parentid",
        "cast(r.timestamp as timestamp) as revision_timestamp", "r.contributor.id as revision_userid", "r.contributor.username as revision_username",
        "r.contributor.ip as revision_userip", "r.model as revision_model", "r.format as revision_format", "r.text._bytes as revision_bytes",
        "r.text._deleted as revision_deleted", "r.sha1 as revision_sha1"
      ).toDF()

    // Run summary query for aggregated values
    val revisionQuery = revisionDf.
      where("page_namespace = 0").
      groupBy("page_id").
      agg( count("*").as("revisions"), min("revision_timestamp").as("start"), countDistinct("revision_userid").as("contributors") )

    // Write to db
    revisionQuery.write.jdbc(sqlUrl, "page_revisions", ix.data.com.COM1.sqlOptions)
  }
}
