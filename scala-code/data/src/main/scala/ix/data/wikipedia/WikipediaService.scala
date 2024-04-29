package ix.data.wikipedia

import de.tudarmstadt.ukp.wikipedia.api._
import ix.util.services2.{ServiceException, ModelService}
import org.apache.commons.configuration.Configuration
import scala.collection.JavaConversions._

object WikipediaService extends ModelService[Wikipedia] {
  val host = config.getString("host", "localhost")
  val user = config.getString("user", "root")
  val password = config.getString("password", "")
  val buffer = config.getInt("buffer", 500)

  def load(name: String, confOpt: Option[Configuration]): Wikipedia = {
    val conf = confOpt.get
    val dbConfig: DatabaseConfiguration = new DatabaseConfiguration
    dbConfig.setHost(conf.getString("host", host))
    dbConfig.setDatabase(conf.getString("database"))
    dbConfig.setUser(conf.getString("user", user))
    dbConfig.setPassword(conf.getString("password", password))
    dbConfig.setLanguage(WikiConstants.Language.valueOf(conf.getString("language")))
    new Wikipedia(dbConfig)
  }

  def iterator(name: String, buffer: Int = buffer): Iterator[Page] = {
    val wiki = get(name)
    new PageIterator(wiki, true, buffer)
  }
}