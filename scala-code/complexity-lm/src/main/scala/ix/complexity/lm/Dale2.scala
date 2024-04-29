package ix.complexity.lm

import java.io.{DataInputStream, FileInputStream, InputStreamReader}

import ix.complexity.lucene3.tokenizer.Tokenizer
import ix.util.services2.{ModelService, Models}
import org.apache.commons.configuration.Configuration

import scala.collection.immutable.{SortedSet, TreeSet}

object Dale2 extends ModelService[SortedSet[String]] {
  def load(name: String, confOption: Option[Configuration]): SortedSet[String] = {
    val conf = confOption.get
    val daleFile = conf.getString("file", "dale.txt")
    val stemmer = conf.getString("stemmer", "english-lower-stem")
    val stream = new InputStreamReader(new DataInputStream(new FileInputStream(daleFile)))
    val tokens = Tokenizer.tokens(stream, stemmer)
    TreeSet(tokens.map(_._2).toSeq:_*)
  }

  def apply(words: Iterable[String], model: String = Models.DEFAULT): Int = {
    many(words, model)
  }

  def one(word: String, model: String = Models.DEFAULT): Boolean = {
    getModel(model).contains(word)
  }

  def many(words: Iterable[String], model: String = Models.DEFAULT): Int = {
    val dale = getModel(model)
    words.map{case s: String => if (dale(s)) 1 else 0}.sum
    //words.map( if (dale(_)) 1 else 0 ).sum
  }
}