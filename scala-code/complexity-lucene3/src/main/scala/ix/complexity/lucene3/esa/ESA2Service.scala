package ix.complexity.lucene3.esa

/**
 * Created by f on 09/09/16.
 */
import gnu.trove.map.hash.TIntDoubleHashMap
import ix.complexity.core.InstanceService
import ix.util.services2.{CleanableService, Models, ServiceException}
import org.apache.commons.configuration.Configuration

object ESA2Service extends InstanceService[ESA2Model, ESA2Instance] with CleanableService {
  def load(name: String, confOption: Option[Configuration]): ESA2Model = {
    val conf = confOption.get
    if (!conf.containsKey("database-file")) throw new ServiceException("No database-file specified for esa model " + name)
    val databaseFile: String = conf.getString("database-file")
    val alpha = conf.getDouble("alpha", .5)
    new ESA2Model(databaseFile, alpha)
  }

  override def load(name: String, conf: Option[Configuration], model: ESA2Model): ESA2Instance = {
    new ESA2Instance(model.file, model.alpha)
  }

  def toSecondOrder(concept: TIntDoubleHashMap, model: String = Models.DEFAULT): TIntDoubleHashMap = {
    val instance = get(model)
    try {
      instance.toSecondOrder(concept)
    } finally {
      offer(model, instance)
    }
  }

  override def cleanUp(): Unit = {
    for(instance <- instances())
      instance.close()
  }
}

case class ESA2Model(file: String, alpha: Double = .5)