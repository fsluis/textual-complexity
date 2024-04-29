package ix.data.com.features

import ix.data.structure.{DataSet, DataSetService}
import ix.util.net2.Servers
import ix.util.net2.spark.SparkDeamon
import ix.util.services2.Services
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

abstract class FeaturesApp extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)
  def appName(dataSet: DataSet): String
  def tableName(dataSet: DataSet): String
  def run(sc: SparkContext, sql: SQLContext, @transient dataSet: DataSet): DataFrame

  override def main(args: Array[String]): Unit = {
    if(args.isEmpty) log.warn("No datasetname supplied as argument, switching to default dataset (phd-10).")
    val dataSetName = args.headOption.getOrElse("ensiwiki_2020-agerank-10")
    val dataSet = DataSetService(dataSetName)

    log.info("Loading Spark/SQL")
    val daemon = SparkDeamon(Servers.localhost).get
    val sc: SparkContext = daemon.context(appName(dataSet))
    val sqlContext = new SQLContext(sc)

    val (sqlUrl, sqlOptions) = ix.data.com.COM1.sqlConfig(dataSetName)
    log.info("Writing to sqlUrl: " + sqlUrl)
    val df = run(sc, sqlContext, dataSet)
    df.write.jdbc(sqlUrl, tableName(dataSet), sqlOptions)

    // Trying to cleanup services (this is just a hack to hopefully get the cleanUp method called on all executors)
    // Seems to work!
    df.rdd.foreachPartition(rows => Services.cleanUp())

    // Closing is needed because of http://stackoverflow.com/questions/28362341/error-utils-uncaught-exception-in-thread-sparklistenerbus
    log.info("Closing Spark")
    daemon.stopSpark()
  }
}
