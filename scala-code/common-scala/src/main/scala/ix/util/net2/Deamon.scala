package ix.util.net2

import org.apache.commons.configuration.HierarchicalConfiguration

trait Deamon {
  def configure(server: Server, conf: HierarchicalConfiguration)
  def name: String
}