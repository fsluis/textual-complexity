package ix.complexity.lucene3.tokenizer

import java.io.{Reader, StringReader}

import ix.util.services2.{Models, ModelService, ServiceException}
import org.apache.commons.configuration.Configuration
import org.apache.lucene3.analysis.tokenattributes.{CharTermAttribute, OffsetAttribute}
import org.apache.lucene3.analysis.{Analyzer => JAnalyzer, TokenStream}

import scala.collection.immutable.ListMap

/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 12/1/13
 * Time: 9:40 PM
 * To change this template use File | Settings | File Templates.
 */
object Tokenizer extends ModelService[JAnalyzer] {

  if (!configurations.contains(Models.DEFAULT))
    put(Models.DEFAULT, Map(
      "language" -> "english"
    ))
  put("english-lower", Map(
    "language" -> "english",
    "lower" -> true
  ))
  put("english", Map(
    "language" -> "english"
  ))
  put("english-lower-stem", Map(
    "language" -> "english",
    "lower" -> true,
    "stop" -> false,
    "stem" -> true
  ))
  put("english-all", Map(
    "language" -> "english",
    "lower" -> true,
    "stop" -> true,
    "stem" -> true,
    "stopWords" -> (System.getProperty("user.home") + "/farm/share/stopwords/english_stop.txt")
  ))

  def load(name: String, confOpt: Option[Configuration]): JAnalyzer = {
    val conf = confOpt.get
    val args = (
      conf.getBoolean("lower", false),
      conf.getBoolean("stop", false),
      conf.getBoolean("stem", false),
      Option(conf.getString("stopWords")))
    Option(conf.getString("language")) match {
      case Some("english") => new Analyzer(args) with English
      case Some("dutch") => new Analyzer(args) with Dutch
      case None => throw new ServiceException("No language set for tokenizer model " + name)
      case unknown => throw new ServiceException("Unknown language " + unknown + " set for tokenizer model " + name)
    }
  }

  //def apply(text: String, model: String = Models.DEFAULT): Iterator[(Int, String)] = 
  //  tokens(text, model)
  def apply(text: String, model: String = Models.DEFAULT, baseOffset:Int = 0): Iterator[(Int,String)] =
    tokens(new StringReader(text), model, baseOffset).map(kv => kv._1.toInt -> kv._2)

  def tokenize(text: String, model: String = Models.DEFAULT, baseOffset:Long = 0): Iterator[(Long,String)] =
    tokens(new StringReader(text), model, baseOffset)

  def tokens(reader: Reader, model: String = Models.DEFAULT, baseOffset:Long = 0): Iterator[(Long,String)] = new Iterator[(Long, String)] {
    val stream: TokenStream = getModel(model).tokenStream("contents", reader)
    val term: CharTermAttribute = stream.addAttribute(classOf[CharTermAttribute])
    val pos: OffsetAttribute = stream.addAttribute(classOf[OffsetAttribute])
    var ready = false
    //ready to read next token?
    var closed = false //is this stream closed?
    stream.reset()

    def hasNext: Boolean = {
      if (closed)
        false
      else if (ready)
        true
      else {
        ready = stream.incrementToken()
        if (!ready) close()
        ready
      }
    }

    def close() {
      // Close stream... http://lucene.apache.org/core/4_6_0/core/index.html?org/apache/lucene/analysis/Analyzer.html
      //println("\nclosing stream of "+model)
      stream.end()
      stream.close()
      closed = true
    }

    def next(): (Long, String) = {
      if (!ready && !hasNext) return (-1, "")
      ready = false
      (baseOffset + pos.startOffset(), term.toString)
      //(pos.startOffset(), term.toString)
    }

    def toListMap: ListMap[Long,String] =
      ListMap(this.toSeq:_*)
  }
}
