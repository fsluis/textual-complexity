package ix.util.net2.jdbc

import ix.util.net2.Server
import org.apache.commons.configuration.HierarchicalConfiguration

/**
 * Created by f on 22/02/18.
 */
object PostgresDaemon {
  val name = "postgres"
}

class PostgresDaemon extends JdbcDaemon {
  def name: String = PostgresDaemon.name
  def driverClassName: String = PostgresDatabase.driverClassName
  // Don't set any properties here, they'll be overwritten (strangely enough) by configure.

  override def configure(server: Server, conf: HierarchicalConfiguration): Unit = {
    super.configure(server, conf)
    // Load driver here/now, during configure phase. The class is loaded before to pre-load the daemons.
    Class.forName(driverClassName).newInstance

    port = properties.getProperty("port", "5432").toInt
    // This one is standard needed for batch statement (++=) using Slick
    // see https://github.com/pgjdbc/pgjdbc/pull/491
    properties.put("reWriteBatchedInserts", "true")
    // Needed for Spark
    // see https://stackoverflow.com/questions/29552799/spark-unable-to-find-jdbc-driver
    properties.put("driver", driverClassName)
    // Needed to automatically "cast" strings to enums
    // doesn't seem to work though
    // https://stackoverflow.com/questions/851758/java-enums-jpa-and-postgres-enums-how-do-i-make-them-work-together
    properties.put("stringtype", "unspecified")
}

  override def newDatabase(name: String): JdbcDatabase =
    new PostgresDatabase(name, this)
}

object PostgresDatabase {
  val driverClassName = "org.postgresql.Driver"
}

class PostgresDatabase(name: String, daemon: PostgresDaemon) extends JdbcDatabase(name,daemon) {
  pool.setDriverClassName("org.postgresql.Driver")
  // See https://jdbc.postgresql.org/documentation/80/connect.html
  def getUrl: String = s"""jdbc:postgresql://${deamon.server.ip}:${deamon.port}/$name"""

}
