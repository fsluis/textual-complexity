package ix.util.services2

import org.apache.commons.configuration.{MapConfiguration, Configuration}
import org.apache.commons.logging.{LogFactory, Log}
import scala.collection.mutable
import scala.collection.JavaConversions._

/**
 * Created by f on 09/06/16.
 */
object Models {
  val DEFAULT = "default"
  private final val log: Log = LogFactory.getLog(this.getClass)
}

/**
 *
 * @tparam M Data model type
 */
trait Models[M] extends Service {
  val models = new mutable.HashMap[String, M] with mutable.SynchronizedMap[String, M]
  val configurations = new mutable.HashMap[String, Configuration]

  for {
    model <- config.configurationsAt("models.model")
    name <- Option(model.getString("name"))
  } yield put(name, model)

    //println("keys: "+config.getKeys.mkString(","))
    //println("model keys: "+config.getKeys("model").mkString(","))
    //println("model keys: "+config.getKeys("model").mkString(","))
    //println("keys:"+config.configurationsAt("models.model").map((configuration: HierarchicalConfiguration) => configuration.getKeys.mkString(",")).mkString(";"))

  def getModel(name: String): M =
    models.getOrElse(name, load(name))

  def getConfig(name: String): Option[Configuration] = {
    if(Models.DEFAULT.equals(name) && !configurations.contains(Models.DEFAULT)) {
      Models.log.info("For model "+name+", found config with "+configurations.size+" keys: "+configurations.keySet.mkString(","))
      // for some reason the sequence of the config is inverse!
      configurations.lastOption.map(_._2)
    } else
      configurations.get(name)
  }

  private def load(name: String): M = {
    val conf = getConfig(name)
    if (conf.isEmpty) Models.log.warn("No config available for model " + name + " on "+this.getClass.getSimpleName)
      //.getOrElse(throw new ServiceException("Unknown model requested: " + name + " on "+this.getClass.getSimpleName))

    //val conf = entry.getOrElse(throw new ServiceException("Unknown model requested: " + name + " on "+this.getClass.getSimpleName))
    synchronized {
      models.getOrElseUpdate(name, {
        Models.log.info("Loading "+getClass.getSimpleName+" model " + name)
        load(name, conf)
      })
    }
    //Models.log.info("Loading "+getClass.getSimpleName+" model " + name)
    //val conf = configurations.get(name).getOrElse(throw new ServiceException("Unknown model requested: " + name + " on "+this.getClass.getSimpleName))
    //models.getOrElseUpdate(name, load(name, conf))
  }

  // So some implementations can still return a Model even if there's no Conf given
  def load(name: String, confOpt: Option[Configuration]): M

  def put(name: String, conf: Configuration): Unit = {
    Models.log.info("For model "+name+" adding keys: "+conf.getKeys.mkString(","))
    configurations.put(name, conf)
  }

  def put(name: String, conf: Map[String,_]): Unit =
    put(name, new MapConfiguration(conf))
}