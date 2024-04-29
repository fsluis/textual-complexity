package ix.util.net2.jdbc

import java.util.Properties

import ix.util.net2.{Deamon, Server, Servers}
import org.apache.commons.configuration.HierarchicalConfiguration

import scala.collection.mutable
import scala.util.Try

/**
 * Created by f on 22/02/18.
 */
abstract class JdbcDaemon extends Deamon {
  var port = 3306
  var properties = new Properties()
  val databases = new mutable.HashMap[String, JdbcDatabase] with mutable.SynchronizedMap[String, JdbcDatabase]
  var server: Server = null

  def configure(server: Server, conf: HierarchicalConfiguration) {
    port = conf.getInt("port", port)
    properties = conf.getProperties("properties", properties)
    //properties.put("driver", driver.getClass.getName)
    this.server = server
  }

  def database(name: String): JdbcDatabase =
    databases.getOrElseUpdate(name, newDatabase(name))

  def newDatabase(name: String): JdbcDatabase
}

object JdbcDaemon {
  def apply(serverName: String, engine: String = MySQLDeamon.name): Option[JdbcDaemon] = {
    for {
      server <- Servers(serverName)
      engine <- server.deamon[JdbcDaemon](engine)
      daemon <- Try{ engine.asInstanceOf[JdbcDaemon] }.toOption
    } yield daemon
  }
}