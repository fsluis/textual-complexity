package ix.util.net2

import java.net.InetAddress
import org.apache.commons.configuration.HierarchicalConfiguration
import org.apache.commons.logging.{LogFactory, Log}

import scala.collection.mutable

/**
 * Represents a server.
 * @author f
 * @version 1.0
 */
class Server(val hostname: String, val ip: String) {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  val address = InetAddress.getByName(ip)
  val deamons = new mutable.HashMap[String, DaemonWrapper]

  def deamon[T <: Deamon](name: String): Option[T] = {
    deamons.get(name).map(_.get()).asInstanceOf[Option[T]]
  }

  private def newDeamon(server: Server, config: HierarchicalConfiguration): Option[Deamon] = {
    try {
      val klass = Option(config.getString("class"))
      val deamon: Deamon = Class.forName(klass.get).newInstance.asInstanceOf[Deamon]
      //deamon.configure(server, config)
      log.info("Registering deamon "+deamon.name+" at server "+server.hostname)
      Some(deamon)
    } catch {
      case e: Exception => log.error("Couldn't load deamon for server "+server.hostname, e)
        None
    }
  }

}

case class DaemonWrapper(server: Server, daemon: Deamon, config: HierarchicalConfiguration, var isConfigured: Boolean) {
  def configure() = {
    daemon.configure(server, config)
    isConfigured = true
  }
  def name = daemon.name
  def get() = {
    if(!isConfigured)
      configure()
    daemon
  }
}
