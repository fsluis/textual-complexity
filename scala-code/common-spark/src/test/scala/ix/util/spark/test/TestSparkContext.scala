package ix.util.spark.test

import org.scalatest._
import scala._
import org.apache.spark.SparkContext
import org.apache.commons.logging.{LogFactory, Log}

// C/p'ed from spark test sources (no way to include these via mvn)

/** Shares a local `SparkContext` between all tests in a suite and closes it at the end */
trait TestSparkContext { self: Suite =>
  val home = sys.env.get("SPARK_HOME")  //spark dir on local drive
  val jar = sys.env.get("SPARK_JAR")
  val uri = sys.env.get("MASTER").getOrElse("local")
  val name = sys.env.get("SPARK_NAME").getOrElse("SparkTest")

  @transient private var _sc: SparkContext = _

  def sc: SparkContext = _sc

  def startSpark() {
    if(home.isDefined && jar.isDefined) {
      TestSparkContext.log.info("Starting Spark with MASTER=\""+uri+"\", SPARK_NAME=\""+name+"\", SPARK_HOME=\""+home.get+"\" and SPARK_JAR=\""+jar.get+"\"")
      _sc = new SparkContext(uri, name, home.get, List(jar.get))
      //_sc.addJar(jar.get)
      //_sc.addJar("http://repo1.maven.org/maven2/commons-jxpath/commons-jxpath/1.3/commons-jxpath-1.3.jar")
    } else {
      TestSparkContext.log.info("Starting Spark with MASTER=\""+uri+"\" and SPARK_NAME=\""+name+"\"")
      _sc = new SparkContext(uri, name)
    }
  }

  def stopSpark() {
    if (_sc != null) {
      TestSparkContext.stop(_sc)
      _sc = null
    }
  }

  def getExecInfo(): IndexedSeq[Map[String,String]] = {
    val storageStatusList = sc.getExecutorStorageStatus
    for (b <- 0 until storageStatusList.size) yield getExecInfo(b)
  }

  private def getExecInfo(a: Int): Map[String, String] = {
    val status = sc.getExecutorStorageStatus(a)
    val execId = status.blockManagerId.executorId
    val hostPort = status.blockManagerId.hostPort
    val rddBlocks = status.blocks.size.toString
    val memUsed = status.memUsed.toString
    val maxMem = status.maxMem.toString
    val diskUsed = status.diskUsed.toString

    Map(
      "execId" -> execId,
      "hostPort" -> hostPort,
      "rddBlocks" -> rddBlocks,
      "memUsed" -> memUsed,
      "maxMem" -> maxMem,
      "diskUsed" -> diskUsed
    )
  }
}

object TestSparkContext {
  private final val log: Log = LogFactory.getLog(this.getClass)

  def stop(sc: SparkContext) {
    sc.stop()
    // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
    System.clearProperty("spark.driver.port")
    System.clearProperty("spark.hostPort")
  }
}