package ix.complexity.stanford

import edu.stanford.nlp.ling.{TaggedWord, Word}
import edu.stanford.nlp.tagger.maxent.{MaxentTagger => JMaxentTagger}
import ix.util.services2.{ModelService, Models}
import org.apache.commons.configuration.Configuration

import scala.collection.JavaConversions._
import scala.collection.immutable.{SortedMap, TreeMap}

object Tagger extends ModelService[JMaxentTagger] {
  def load(name: String, confOpt: Option[Configuration]): JMaxentTagger = {
    val taggerPath = confOpt.get.getString("model", "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger")
    new JMaxentTagger(taggerPath)
  }

  def apply(sentence: SortedMap[Int,String], model: String = Models.DEFAULT): SortedMap[Int,TaggedWord] = {
    // Preprocess to stanford format
    //val words = sentence.zipWithIndex.map{case ((start: Int, token: String), i: Int) => (i, new Word(token,start,start+token.length))}

    val words = for(((start,token),i) <- sentence.zipWithIndex)
      yield (i, new Word(token,start,start+token.length))
    //System.out.println("Words: "+words)
    //System.out.println("Words: "+words.values.toList)

    val tagger = getModel(model)
    //ListMap(tagger.apply(words.values.toList).toSeq:_*)
    TreeMap(tagger.apply(words.values.toList).toList.zipWithIndex.map(_.swap).toSeq:_*)
  }
}