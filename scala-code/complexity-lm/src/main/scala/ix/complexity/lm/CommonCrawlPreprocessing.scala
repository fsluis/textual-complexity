package ix.complexity.lm

import ix.common.util.TimeLogger
import ix.complexity.core.InstanceService
import ix.complexity.lm.KenLM.config
import ix.util.services2.{CleanableService, ServiceException}
import net.liftweb.json.JsonParser.ParseException
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import net.liftweb.json._

import scala.util.matching.Regex

/**
 * Created by f on 11/08/16.
 *
 * For some reason this doesn't do any multithreading ....
 */
object CommonCrawlPreprocessing extends InstanceService[CommonCrawlModel, CommonCrawlPreprocessingInstance] with CleanableService {
  val log: Log = LogFactory.getLog(classOf[CommonCrawlModel])
  val tlog1: TimeLogger = new TimeLogger(log, "prepro parsing and waiting", 1000)
  val waitForInput: Int = 100
  val initTimeLimit: Int = 30 * 60 * 1000
  val preproHome = config.getString("commoncrawl-prepro-home", System.getProperty("user.home") + "/farm/src/buck2014/prepro/")
  val name = "model"
  // Note: Removed this! April2020. Make sure to use full path again.
  //val modelsHome: String = config.getString("models-home", "/local/farm/data/com1/languagemodels")
  implicit val formats = new DefaultFormats {}

  override def load(name: String, confOpt: Option[Configuration]): CommonCrawlModel = {
    CommonCrawlModel(name)
  }

  override def load(name: String, conf: Option[Configuration], model: CommonCrawlModel): CommonCrawlPreprocessingInstance = {
    val instance = new CommonCrawlPreprocessingInstance()
    instance.init()
    log.info("Done initializing Prepro instance with pid "+instance.getPid+", returning "+instance)
    instance
  }

  override def cleanUp(): Unit = {
    for(instance <- instances())
      instance.cleanUp()
  }

  def text(text: String): Seq[String] = {
    // the prepro works per newline - thus we need to split the text on newlines and later combine the sentences again
    text.split("\n").
      map(_.trim).
      filter(_.nonEmpty).
      map(singleLine).
      foldLeft(Seq.empty[String])( _++_ ) // foldLeft preserves order and can work on an empty list (cause of its starting value)
  }

  val p_start: Regex = """^<P>""".r
  val p_end: Regex = """</P>$""".r
  def singleLine(line: String): Seq[String] = {
    val parsing = tlog1.start()
    val irstlm = get(name) // placeholder name
    //log.info("Got IRSTLM instance: "+irstlm)
    val option = try {
      irstlm.doReduce(line, 0)
    } finally {
      //log.info("Returning IRSTLM instance: "+irstlm)
      offer(name, irstlm)
    }
    tlog1.stop(parsing)
    option.
      map(_.trim).
      map(p_start.replaceFirstIn(_, "")).
      map(p_end.replaceFirstIn(_, "")).
      map(_.split("\n").toSeq).
      getOrElse(Seq.empty[String]).
      filter(_.trim.nonEmpty) // because the <P> and </P> removal leaves empty strings at beginning and end of the list
  }

  def stopInstances(name: String) = {
    cleanUp(name, (instance: CommonCrawlPreprocessingInstance) => instance.cleanUp())
  }
}

case class CommonCrawlModel(name: String)

/*
import ix.complexity.lm.CommonCrawlPreprocessing
CommonCrawlPreprocessing("This is a test sentence. And another sentence.")
*/