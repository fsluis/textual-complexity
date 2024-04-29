package ix.util.services2

import org.apache.commons.configuration.HierarchicalConfiguration

/**
 * Main Service trait. Inherit this trait to register your Scala singleton as a Service. This will make
 * available an instance of HierarchicalConfiguration named 'config'.
 * Only use this trait in a singleton object to assure at maximum one Service is running per JVM.
 * User: f
 * Date: 12/1/13
 * Time: 4:03 PM
 */
trait Service {
  val config = Services.register(this)
  startService(config)

  @Deprecated
  def startService(config :HierarchicalConfiguration) = {}

  @Deprecated
  def stopService() = ()
}
