package ix.util.spark

import ix.util.net2.spark.SparkDeamon
import ix.util.services2.Service
import org.apache.commons.configuration.HierarchicalConfiguration
import ix.util.net2.Servers
import scala.util.Try
import org.apache.spark.SparkContext

/**
 * @deprecated. Use SparkDeamon companion object instead.
 */
object SparkService extends Service {
  val server = config.getString("server", "local")
  val deamon: SparkDeamon = findDeamon.getOrElse(new SparkDeamon)

  private def findDeamon = for {
      server <- Servers(server)
      deamon <- server.deamon[SparkDeamon](SparkDeamon.name)
    } yield deamon

  def context: SparkContext =
    deamon.getContext

  def context(name: String): SparkContext = {
    deamon.sparkName = name
    context
  }
}