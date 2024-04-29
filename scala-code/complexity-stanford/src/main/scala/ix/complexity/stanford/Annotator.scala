package ix.complexity.stanford

import java.util.Properties
import edu.stanford.nlp.pipeline.{Annotation, CoreDocument, StanfordCoreNLP}
import ix.common.util.TimeLogger
import ix.complexity.core.InstanceService
import ix.util.services2.ModelService
import org.apache.commons.configuration.{Configuration, ConfigurationConverter}
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import scala.collection.immutable

/**
 * Main StanfordCoreNLP interface. Use this to setup parsing pipelines and annotate documents with it.
 * See https://stanfordnlp.github.io/CoreNLP/memory-time.html for tips on reducing memory/time load.
 */
object Annotator extends InstanceService[Properties, StanfordCoreNLP] {
  val log: Log = LogFactory.getLog(this.getClass)
  val tlog = new TimeLogger(log, "stanford parsing", 5000)

  // Note about using dots/periods in key names @ http://commons.apache.org/proper/commons-configuration/userguide/howto_hierarchical.html#Escaping_special_characters
  override def load(name: String, confOpt: Option[Configuration]): Properties = {
    val props = confOpt.map(ConfigurationConverter.getProperties).getOrElse(new Properties)
    props.stringPropertyNames().filter(_.contains("..")).
      foreach(name => props.setProperty(name.replace("..", "."), props.getProperty(name)))
    props
  }

  override def load(name: String, conf: Option[Configuration], props: Properties): StanfordCoreNLP = new StanfordCoreNLP(props)

  /**
   * Runs Stanford's pipeline to text, given a particular model / settings in services.xml
   * Note that this method can throw an exception! The exceptions occuring in the Stanford Pipeline are _not_ caught here.
   * Note that CoreDocument and all its associated classes are _not_ serializable.
   * @param text
   * @param modelName
   * @return CoreDocument
   */
  def apply(text: String, modelName: String): CoreDocument = {
    val instance = get(modelName)
    val document = new CoreDocument(text)
    try {
      val t = tlog.start()
      // model.process(text) has memory leak? See https://stackoverflow.com/questions/48683125/stanford-corenlp-memory-leak
      // in the comments there it points to thread-safety issues that might cause the leak
      instance.annotate(document)
      //StanfordCoreNLP.clearAnnotatorPool(); < not needed!
      tlog.stop(t)
    } finally {
      offer(modelName, instance)
    }
    document
  }
}
