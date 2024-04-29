package ix.complexity.lucene3.tokenizer

import org.tartarus.snowball3.SnowballProgram
import org.tartarus.snowball3.ext.EnglishStemmer

trait English extends Analyzer {
  // part of https://svn.apache.org/repos/asf/lucene/dev/branches/fieldtype/modules/analysis/common/src/resources/org/apache/lucene/analysis/snowball/
  def defaultStopWords: String = "english_stop.txt"
  def stemmer: SnowballProgram = new EnglishStemmer()
}
