package ix.complexity.lucene3.tokenizer

import org.apache.lucene3.analysis.{Tokenizer => JTokenizer, _}
import java.io.{File, Reader}
import org.apache.lucene3.analysis.standard.{StandardFilter, StandardTokenizer}
import org.apache.lucene3.analysis.snowball.SnowballFilter
import org.apache.lucene3.util.IOUtils
import org.apache.lucene3.util.Version.{LUCENE_36 => version}
import org.tartarus.snowball3.SnowballProgram
import org.apache.lucene3.analysis.ReusableAnalyzerBase.TokenStreamComponents
import scala.Some

abstract class Analyzer (
  val lower :Boolean = false,
  val stop: Boolean = false,
  val stem: Boolean = false,
  val stopWords: Option[String] = None)
  extends ReusableAnalyzerBase {

  // Helper method to be able to use a preset tuple of arguments
  def this(x: (Boolean,Boolean,Boolean,Option[String])) = this(x._1, x._2, x._3, x._4)

  // Taken from org/apache/lucene/analysis/nl/DutchAnalyzer.java
  def createComponents(fieldName: String, aReader: Reader): TokenStreamComponents = {
    val source: JTokenizer = new StandardTokenizer(version, aReader)
    val filters = snowballFilter(stopFilter(lowerCaseFilter(new StandardFilter(version, source))))
    new TokenStreamComponents(source, filters)
  }

  private def lowerCaseFilter(prev: TokenStream) = 
    if(lower) new LowerCaseFilter(version, prev) else prev
  
  private def snowballFilter(prev: TokenStream) =
    if(stem) new SnowballFilter(prev, stemmer) else prev
  
  def stemmer: SnowballProgram

  private def stopFilter(prev: TokenStream) =
    if(stop) new StopFilter(version, prev, loadStopWords) else prev

  private def loadStopWords: CharArraySet = {
    val reader = stopWords match {
      case Some(filename) => IOUtils.getDecodingReader(new File(filename), IOUtils.CHARSET_UTF_8)
      case None => IOUtils.getDecodingReader(classOf[SnowballFilter], defaultStopWords, IOUtils.CHARSET_UTF_8)
    }
    WordlistLoader.getSnowballWordSet(reader, version)
  }

  def defaultStopWords: String
}
