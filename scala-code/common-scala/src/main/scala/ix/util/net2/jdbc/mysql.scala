package ix.util.net2.jdbc

import ix.util.net2.Server
import org.apache.commons.configuration.HierarchicalConfiguration
import scala.reflect._


object MySQLDeamon {
  val name = "mysql"
}

class MySQLDeamon extends JdbcDaemon {
  def name: String = MySQLDeamon.name
  def driverClassName: String = MySQLDatabase.driverClassName
  // Don't set any properties here, they'll be overwritten (strangely enough) by configure.

  override def configure(server: Server, conf: HierarchicalConfiguration): Unit = {
    super.configure(server, conf)
    // Load driver here/now, during configure phase. The class is loaded before to pre-load the daemons.
    Class.forName(driverClassName).newInstance

    // This one is standard needed for batch statement (++=) using Slick
    // see https://github.com/slick/slick/issues/1272
    properties.put("rewriteBatchedStatements", "true")
    // Needed to prevent memory leaks ...
    // see https://stackoverflow.com/questions/5765990/java-mysql-jdbc-memory-leak
    //properties.put("dontTrackOpenResources", "true")
    // Needed for Spark
    // see https://stackoverflow.com/questions/29552799/spark-unable-to-find-jdbc-driver
    properties.put("driver", driverClassName)
  }

  override def newDatabase(name: String): JdbcDatabase =
    new MySQLDatabase(name, this)
}

object MySQLDatabase {
  val driverClassName = "com.mysql.cj.jdbc.Driver"  // com.mysql.jdbc.Driver "org.gjt.mm.mysql.Driver"
  def apply(databaseName: String, serverName: String): Option[MySQLDatabase] = JdbcDatabase(databaseName, serverName)(classTag[MySQLDatabase])
}

class MySQLDatabase(override val name: String, override val deamon: MySQLDeamon) extends JdbcDatabase(name,deamon) {
  pool.setDriverClassName("org.gjt.mm.mysql.Driver")

  def getUrl: String = s"""jdbc:mysql://${deamon.server.ip}:${deamon.port}/$name"""
}

