package ix.complexity.lm

import ix.common.util.Utils
import java.lang

import ix.util.services2.{CleanableService, ModelService, Models}
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import scala.reflect.io.File

object WordFrequencies extends ModelService[FrequenciesModel] with CleanableService {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  def load(name: String, confOpt: Option[Configuration]): FrequenciesModel = {
    //val sqlFile = confOpt.get.getString("file", "frequencies.db")
    val filename = confOpt.get.getString("file", "frequencies.db")
    val tmpFile = Utils.copyFileToTemp(filename)
    log.info(s"Copied $filename to local copy ${tmpFile.toString}")
    val model = new FrequenciesModel(tmpFile.toString)
    model.init()
    model
  }


  override def cleanUp(): Unit = {
    for( (name, model) <- models) {
      log.info(s"Closing ${name}, deleting ${model.getDatabaseFile}")
      model.close()
      File(model.getDatabaseFile).delete()
    }
  }

  def apply(words: Iterable[String], model: String = Models.DEFAULT): List[Double] = {
    many(words, model)
  }

  def many(words: Iterable[String], model: String = Models.DEFAULT): List[Double] = {
    getModel(model).getFrequencies(words.toIterator).toList.map((d: lang.Double) => d.toDouble )
  }

  def query(words: Iterator[String], model: String = Models.DEFAULT): TraversableOnce[Double] = {
    getModel(model).getFrequencies(words).map((d: lang.Double) => d.toDouble )
  }
}