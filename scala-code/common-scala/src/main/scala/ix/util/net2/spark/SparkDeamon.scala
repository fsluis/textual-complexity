package ix.util.net2.spark

import ix.util.net2.{Deamon, Server, Servers}
import org.apache.commons.configuration.HierarchicalConfiguration
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.JavaConversions._

object SparkDeamon {
  private val log: Log = LogFactory.getLog(this.getClass)
  val name = "spark"

  def apply(serverName: String = "local"): Option[SparkDeamon] = {
    for {
      server <- Servers(serverName)
      deamon <- server.deamon[SparkDeamon](SparkDeamon.name)
    } yield deamon
  }
}

class SparkDeamon extends Deamon {
  var home: Option[String] = None
  var jars: List[String] = List()
  var uri = "local"
  var sparkName = "SparkTest"
  //var _sc: SparkContext = null
  // update to spark > 2.0.0. SparkContext is no longer the way to start Spark programmatically.
  var _session: Option[SparkSession] = None
  var secret: Option[String] = None
  var memoryFraction: Option[String] = None
  // doesn't work
  //var propertiesFile: Option[String] = None

  def configure(server: Server, conf: HierarchicalConfiguration): Unit = {
    SparkDeamon.log.debug("Spark conf keys: "+conf.getKeys.mkString(", "))
    home = Option(sys.env.getOrElse("SPARK_HOME", conf.getString("spark-home")))
    jars = conf.getList("spark-jars").map(_.toString).toList ::: jars
    if (sys.env.contains("ADD_JARS")) jars = sys.env("ADD_JARS") :: jars
    uri = sys.env.getOrElse("MASTER", Option(conf.getString("spark-uri")).getOrElse(uri))
    sparkName = sys.env.getOrElse("SPARK_NAME", Option(conf.getString("spark-name")).getOrElse(sparkName))
    secret = Option(conf.getString("spark-secret"))
    memoryFraction = Option(conf.getString("memory-fraction"))
    //propertiesFile = Option(conf.getString("spark-properties"))
    println("uri "+uri+"; jars "+jars.mkString(",")+"; home "+home)
  }

  def name: String = SparkDeamon.name

  def startSpark(): SparkSession = {
    if (jars.isEmpty) SparkDeamon.log.warn("No JARS set for Spark! ($ADD_JARS)")
    if(home.isDefined && jars.size>0) {
      //println(sys.env.keys)
      val conf: SparkConf = new SparkConf()
      conf.setSparkHome(home.get)
      conf.setJars(jars)
      val envVars = sys.env.filter((t: (String, String)) => t._1.toLowerCase.startsWith("spark")).toMap
      conf.setExecutorEnv(envVars.toArray)
      if(memoryFraction.isDefined) conf.set("spark.shuffle.memoryFraction", memoryFraction.get)
      //propertiesFile.foreach( conf.set("spark.application.properties.file", _) )
      if(secret.isDefined) {
        conf.set("spark.authenticate", "true")
        conf.set("spark.authenticate.secret", secret.get)
        SparkDeamon.log.info("Using spark secret: "+secret.get)
      } else
        SparkDeamon.log.info("Warning: no secret-auth user")
      SparkDeamon.log.info("Starting Spark with MASTER=\""+uri+"\", SPARK_NAME=\""+sparkName+"\", SPARK_HOME=\""+home.get+"\" and SPARK_JAR=\""+jars.mkString(",")+"\" and "+envVars.size+" environment variables ("+envVars.keys.mkString(",")+")")
      //_sc = new SparkContext(uri, sparkName, conf)
      SparkSession.builder().appName(sparkName).master(uri).config(conf).getOrCreate()
    } else {
      SparkDeamon.log.info("Starting Spark with MASTER=\""+uri+"\" and SPARK_NAME=\""+sparkName+"\". Warning: No secret-auth used. Warning: SPARK_JAR was not defined.")
      //_sc = new SparkContext(uri, sparkName)
      SparkSession.builder().appName(sparkName).master(uri).getOrCreate()
    }
  }

  def stopSpark() {
    _session.foreach(_.stop)
    _session = None
    // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
    System.clearProperty("spark.driver.port")
    System.clearProperty("spark.hostPort")
  }

  def context(sparkName: String = "SparkTest"): SparkContext = {
    this.sparkName = sparkName
    getContext
  }

  def getSession: SparkSession = {
    if(_session.isEmpty)
      _session = Some(startSpark())
    _session.get
  }

  def session(sparkName: String = "SparkTest"): SparkSession = {
    this.sparkName = sparkName
    getSession
  }

  def getContext: SparkContext = {
    getSession.sparkContext
  }
}