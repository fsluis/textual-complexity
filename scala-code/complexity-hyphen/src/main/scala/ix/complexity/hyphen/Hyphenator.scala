package ix.complexity.hyphen

import java.io.File

import ch.sbs.jhyphen.{Hyphenator => JHyphenator, Hyphen}
import ix.util.net2.Servers
import ix.util.services2.{ServiceException, ModelService, Models}
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}

object Hyphenator extends ModelService[JHyphenator] {

  /*put("english", Map(
    "url" -> "resources/hyphen/hyph_en_US.dic"
  ))
  put("dutch", Map(
    "url" -> "resources/hyphen/hyph_nl_NL.dic"
  ))*/
  //if (!configurations.contains(Models.DEFAULT))
  //  put(Models.DEFAULT, configurations("english"))
  System.setProperty("jna.nosys", "true")

  lazy val log: Log = LogFactory.getLog(this.getClass)
  val SEP = '-'

  def load(name: String, confOpt: Option[Configuration]): JHyphenator = {
    new JHyphenator(new File(confOpt.get.getString("url")))
  }

  def hyphenate(word: String, model: String = Models.DEFAULT): String = {
    val res = try {
      get(model).hyphenate(word, SEP, SEP)
    } catch {
      case e: Throwable =>
        log.error("Error while getting syllables for word: "+word, e)
        word
    }
    //log.info("hyphens of "+word+": "+res)
    res
  }

  def syllables(word: String, model: String = Models.DEFAULT): Array[String] = {
    //log.info("Getting syllables of word \""+word+"\"")
    val res = hyphenate(word, model).split(SEP)
    //log.info("Syllables of word \""+word+"\""+": "+res)
    // blabla
    res
  }
}
