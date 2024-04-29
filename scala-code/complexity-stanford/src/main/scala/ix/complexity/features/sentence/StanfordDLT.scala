package ix.complexity.features.sentence

import java.util

import edu.stanford.nlp.ling.{CoreLabel, TaggedWord}
import edu.stanford.nlp.pipeline.CoreSentence
import edu.stanford.nlp.trees.{GrammaticalStructure, TypedDependency}
import ix.complexity.stanford.Dependency

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.immutable.SortedMap

trait StanfordLocality {
  /**
   * Helper function for the new Stanford API
   * @param sentence CoreSentence
   * @param dltPos pos tags to count (leave empty to include all in the count)
   * @param collapsePhrases whether to collapse repeated consequitive POS tags and count as 1
   * @param excludeEndings whether to exclude the ending/last word from the count
   * @return locality count per dependency in sentence
   */
  def locality(sentence: CoreSentence, dltPos: List[String] = List(), collapsePhrases: Boolean = false, excludeEndings: Boolean = false): List[Int] = {
    val tokens = sentence.tokens()
    val graph = sentence.dependencyParse()
    val dependencies = graph.typedDependencies().asScala.toList
    // the index-1 is just to compensate for the index+1 in calculateLocality
    val tagged = tokens.map(tok => tok.index()-1 -> new TaggedWord(tok.word(), tok.tag()))
    val sorted = SortedMap( tagged:_* )
    // the cases is to be in line with the Dependencies class
    val cases = dependencies.map(dep => Dependency(dep.dep(), dep.gov(), dep.reln().getShortName) )
    // use old method to do the calculations
    calculateLocality(sorted, cases, dltPos, collapsePhrases, excludeEndings)
  }

  def calculateLocality(tagged: SortedMap[Int, TaggedWord], dependencies: List[Dependency], dltPos: List[String] = List(), collapsePhrases: Boolean = false, excludeEndings: Boolean = false): List[Int] = {
    //val dependencies = grammar.typedDependenciesCollapsed()
    //System.out.println("Dependencies: "+dependencies)
    //System.out.println("Tokens: "+tokens)
    //System.out.println("Tagged: "+tagged)
    dependencies.flatMap((dependency: Dependency) => {
      val left = math.min(dependency.gov.index, dependency.dep.index)
      val right = math.max(dependency.gov.index, dependency.dep.index)
      if (left>0 && right>0) { // to exclude the root node, which is not a word
        //val dist = right-left
        // the range is +1 to make it inclusive for the last token
        // the tagged.get(i-1) is -1 because there is a misalignment between the tagged map (starting at 0) and the
        // dependencies (starting at 1).
        val words = Iterator.range(left, right+1).map(i => tagged.get(i-1) ).toList.flatten
        val dist = calculateDistance(words, dltPos, collapsePhrases, excludeEndings)
        //if(dlt>0)
        //System.out.println("Dep "+dependency.toString+"; dist: "+dist+"; words: "+words+"; range:"+Iterator.range(left, right).toList)
        Some(dist)
      } else None
    }).toList
  }

  /**
   * Options:
   * @param words
   * @param dltPos Only count specific POS tags
   * @param collapsePhrases Count consecutive pos tags as one
   * @param excludeEndings Count first and last token or not (better not/true: Every dependency starts at length two otherwise)
   * @return dependency length
   */

  def calculateDistance (words: List[TaggedWord], dltPos: List[String] = List(), collapsePhrases: Boolean = false, excludeEndings: Boolean = false) = {
    var previous: String = ""

    words.zipWithIndex.map{case (word: TaggedWord, i: Int) =>
      val pos = truncatePosTag(word)
      val collapse = (collapsePhrases && previous == pos)
      val include = dltPos.isEmpty || dltPos.contains(pos)
      val ending = excludeEndings && ( i==0 || i==(words.size-1) )
      //val count = (collapsePhrases && previous != pos) && (dltPosEmptyOrContains(pos, dltPos))
      previous = pos
      if (!collapse && include && !ending) 1 else 0
    }.sum
  }

  /**
   * Small trick to group the list of POS tags
   */
  def truncatePosTag(word: TaggedWord): String = {
    word.tag.slice(0, 2) match {
      case "NN" => "NN"
      case "VB" => "VB"
      case _ => word.tag
    }
  }


}

object StanfordDLT extends StanfordLocality {
  /**
   * @deprecated Use calculateLocality instead
   * PhD settings:
   * dltPos = List("NN", "VB")
   * excludeEndings = true (but slightly different definition! In PhD the first token was not excluded)
   * collapse = true
   */
  def apply(tagged: SortedMap[Int, TaggedWord], dependencies: List[Dependency], dltPos: List[String] = List("NN", "VB")): List[Int] =
    calculateLocality(tagged, dependencies, dltPos, collapsePhrases = true, excludeEndings = true)
}