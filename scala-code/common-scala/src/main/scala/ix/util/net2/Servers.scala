package ix.util.net2

import ix.util.services2.Service
import org.apache.commons.configuration.HierarchicalConfiguration
import scala.collection.mutable
import java.net.InetAddress
import org.apache.commons.logging.{LogFactory, Log}
import scala.collection.JavaConversions._

object Servers extends Service {
  lazy val log: Log = LogFactory.getLog(this.getClass)
  lazy val localhost = InetAddress.getLocalHost.getHostName
  lazy val servers = new mutable.HashMap[String, Server] {
    override def get(key: String) = if (key=="local") super.get(localhost) else super.get(key) }

  def local() = servers.get(localhost)

  def apply(name: String = localhost): Option[Server] = {
    val res = servers.get(name)
    if(res.isEmpty)
      log.info("Requested server "+name+" not found")
    res
  }

  override def startService(config: HierarchicalConfiguration) {
    super.startService(config)
    for {
      server <- config.configurationsAt("servers.server")
      hostname <- Option(server.getString("hostname"))
    } yield servers += hostname -> newServer(hostname, server)
    // Todo: move outside of startService method
  }

  private def newServer(hostname: String, config: HierarchicalConfiguration) = {
    log.info("Registering server at "+hostname)
    val server = new Server(
      hostname,
      config.getString("address", "127.0.0.1")
    )
    for {
      deamonConfig <- config.configurationsAt("deamons.deamon")
      deamon <- newDeamon(server, deamonConfig)
      // Todo: Make this lazy
    } yield server.deamons += deamon.name -> deamon
    server
  }

  private def newDeamon(server: Server, config: HierarchicalConfiguration): Option[DaemonWrapper] = {
    try {
      val klass = Option(config.getString("class"))
      val deamon: Deamon = Class.forName(klass.get).newInstance.asInstanceOf[Deamon]
      //deamon.configure(server, config)
      log.info("Registering deamon "+deamon.name+" at server "+server.hostname)
      Some(DaemonWrapper(server, deamon, config, false))
    } catch {
      case e: Exception => log.error("Couldn't load deamon for server "+server.hostname, e)
      None
    }
  }
}