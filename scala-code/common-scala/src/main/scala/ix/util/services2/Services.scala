package ix.util.services2

import java.io.File
import java.net.URL
import java.util.ServiceLoader

import org.apache.commons.configuration.{HierarchicalConfiguration, XMLConfiguration}
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, SynchronizedBuffer}

/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 12/1/13
 * Time: 4:02 PM
 * To change this template use File | Settings | File Templates.
 */
object Services {
  val uri = "services.xml"
  val log: Log = LogFactory.getLog(this.getClass)
  // services-list (to retrieve or close later-on)
  val services = new ArrayBuffer[Service] with SynchronizedBuffer[Service]
  val configurations = new mutable.HashMap[String,HierarchicalConfiguration]
  // settings (for each service a settings node)
  val config = getConfiguration
  var cleaned = false

  def register(service :Service): HierarchicalConfiguration = {
    services += service
    //val query = "service[class=\'"+service.getClass.getName+"\']"
    //val subs = config.configurationsAt(query)
    val subs = for {
      sub <- config.configurationsAt("service")
      klass <- Option(sub.getString("class"))
      if klass.equals(service.getClass.getName)
    } yield sub
    // Note: when xpath is used, the keys change from being seperated by a . to a /

    val sub = if(subs.isEmpty) {
      log.warn("Registering service "+service.getClass.getName+" without config")
      new HierarchicalConfiguration()
    } else {
      if(subs.length>1) log.warn("Found multiple setting for "+service.getClass.getName)
      log.info("Registering service "+service.getClass.getName+" with config")
      subs.head
    }

    configurations += service.getClass.getName -> sub
    sub
  }

  def cleanUp(): Unit = {
    this.synchronized {
      if(!cleaned) {
        log.info("Cleaning cleanable services")
        for(service <- services)
          service match {
            case cleanable: CleanableService => cleanable.cleanUp()
            case _ =>
          }
      }
      cleaned = true
    }
  }

  private def getConfiguration: HierarchicalConfiguration = {
    val config = new XMLConfiguration()

    // Todo: @Spark executors, the next line makes it crash with a ClassDefNotFoundException
    // (even though dependencies are included!)
    // maybe try this: http://stackoverflow.com/questions/687071/what-is-the-root-package-in-scala
    //config.setExpressionEngine(new XPathExpressionEngine())

    // First: load jar settings
    // Second: merge with local settings (file location indicated via environment variable)
    val sources = jarSources ::: localSources
    for(u <- sources) config.load(u)
    // Third: load settings from other SettingsProviders (of new jars; extensibility feature)
    val loader = ServiceLoader.load(classOf[SettingsProvider])
    for(p <- loader) config.addNodes(null, List(p.getConfiguration.getRootNode))
    // print stuff
    log.info("Loaded services config from urls ["+sources.mkString(", ")+"] and loaders ["+loader.map((p: SettingsProvider) => p.getClass.getSimpleName).mkString(", ")+"]")
    // Finalize
    config
  }

  private def jarSources: List[URL] = {
    // Resources in jar
    val streams = List(
      Option(ClassLoader.getSystemResource(uri)),
      Option(this.getClass.getResource(uri)),
      Option(Thread.currentThread.getContextClassLoader.getResource(uri)))
    streams.flatten
  }

  private def localSources: List[URL] = {
    // Resources locally
    val locations = List(
      Option(uri),
      Option("conf/"+uri),
      Option("resources/"+uri),
      sys.env.get("SERVICESXML"),
      sys.env.get("SPARK_SERVICES"),
      // An ugly hack ...
      //Option("/local/workspace/ix/ix-api/resources/"+uri),
      Option(System.getProperty("services.xml")))
    locations.
      flatten.
      map((loc: String) => new File(loc)).
      filter((f: File) => f.exists()).
      map((f: File) => f.toURI.toURL)
    // the latter map phase is a bit ridiculous imo but recommended @
    // http://stackoverflow.com/questions/6098472/pass-a-local-file-in-to-url-in-java
  }
}
