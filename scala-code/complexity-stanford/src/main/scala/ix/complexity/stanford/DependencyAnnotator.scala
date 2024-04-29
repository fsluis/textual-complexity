package ix.complexity.stanford

import java.util

import edu.stanford.nlp.ling.{CoreLabel, IndexedWord, TaggedWord}
import edu.stanford.nlp.trees.Tree
import ix.util.ScalaHelper
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.immutable.SortedMap
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Try
/**
 * Version 2 that uses dependencies + phrases for calculating locality
 */
object DependencyAnnotator {
  val log: Log = LogFactory.getLog(this.getClass)
  implicit def coreLabel2Token(label: CoreLabel): Token = Token(label.index(), label.word(), label.tag())

  def apply(text: String, index: Long = 0, debug: Boolean = false): Seq[(Int, Sentence)] = {
    val doc = try {
      Annotator(text, "treeparse") // treeparse+depparse
      // probably, treeparse gives better (or equal) dependencies as depparse
      // depparse is the "fast dependency parser", which does not need the slow treeparse
      // ergo: for speed, not for accuracy
    } catch { case e: Exception =>
      log.warn("Failed while trying to treeparse text", e)
      return Seq.empty[(Int,Sentence)]
    }

    for (sen <- doc.sentences()) yield {

      // tokens
      val tokens = sen.tokens().map(coreLabel2Token).toIndexedSeq

      // noun/verb phrases
      val tree = sen.constituencyParse()
      val phraseTrees = tree.filter(d => d.depth==2 || (d.depth==1 && d.label().value().startsWith("VB")) ) //filter(sub => sub.label().value().endsWith("VP") || sub.label().value().endsWith("NP") ).
      val treePhrases = for(p <- phraseTrees) yield {
        val phraseLabels = p.`yield`(new util.ArrayList[CoreLabel])
        if(debug) println(s"Phrase ${p.label()} @${p.depth}: ${p.getLeaves()}")
        Phrase(p.label().value(), phraseLabels.map(coreLabel2Token).toIndexedSeq, p.depth)
      }

      // dependencies
      val graph = sen.dependencyParse()
      val dependencies = graph.typedDependencies().
        map(dep => IndexedDependency(dep.dep().backingLabel(), dep.gov().backingLabel(), dep.reln().getShortName) ).
        toList

      index + sen.charOffsets().first -> Sentence(tokens, treePhrases, dependencies)
    }
  }
}

abstract class IndexRange(val left: Int, val right: Int) extends Range.Inclusive(start = left, end = right, step = 1) {
  def overlaps(other: IndexRange): Boolean = (other.left <= this.left && this.left <= other.right) || (other.left <= this.right && this.right <= other.right)
}

case class Phrase(label: String, tokens: IndexedSeq[Token], depth:Int)
  extends IndexRange(left = tokens.head.index, right = tokens.last.index) {
  def isVerb: Boolean = label.startsWith("VB")
  def isVerbPhrase: Boolean = label.endsWith("VP")
  def isNounPhrase: Boolean = label.endsWith("NP")
  def is(types: Seq[String]): Boolean = types.exists {
    case "VB" => label.startsWith("VB") // verbs
    case "VP" => label.endsWith("VP") // verb phrase
    case "NP" => label.endsWith("NP") // noun phrase
    case "all2" => depth==2 // all second-level tags
    case "all" => true // all
    case _ => false
  }
}

case class IndexedDependency(dep: Token, gov: Token, relation: String)
  extends IndexRange( left = math.min(gov.index, dep.index), right = math.max(gov.index, dep.index) ) {
  // does not include root node (start of sentence-token), as (almost?) every word has a dependency on root
  def notRoot: Boolean = left > 0 && right > 0
}

case class Sentence(tokens: IndexedSeq[Token], phrases: Seq[Phrase], dependencies: Seq[IndexedDependency])
  extends IndexRange(left = tokens.head.index, right = tokens.last.index) {
  def locality(correction: Int = 2, debug: Boolean = false): Seq[(Int,Int)] = {
    if(debug) println(tokens)
    for (dep <- dependencies.filter(_.notRoot)) yield {
      val overlappingPhrases = phrases.filter( _.overlaps(dep) ) // phrases covered by dep
      val toCollapse = overlappingPhrases.map( _.toSet ).foldLeft(Set[Int]())(_ ++ _) // indexes belonging to phrases
      val indexes = dep.toSet diff toCollapse // indexes not belonging to phrases
      val locality = math.max(indexes.size + overlappingPhrases.size - correction, 0)
      if (debug) {
        val depTokens = mutable.LinkedHashMap( dep.map( i => i -> (tokens(i-1).word+"-"+i) ).sortBy(_._1) :_* )
        overlappingPhrases.filter(p => depTokens.contains(p.left)).foreach( p => depTokens(p.left) = "["+depTokens(p.left) )
        overlappingPhrases.filter(p => depTokens.contains(p.right)).foreach( p => depTokens(p.right) = (depTokens(p.right) + "](" + p.label+"@"+p.left+"--"+p.right+")") )
        println(s"Dep ${dep.left}--${dep.right} loc $locality (Nt=${indexes.size}, Np=${overlappingPhrases.size}): "+ depTokens.values.mkString(" ") )
      }
      (locality, overlappingPhrases.size)
      // in case the dep is within a phrase, the size is 1
      // in case the dep is between two adjacent phrases/tokens, the size is 2
      // in all other cases: more than 2
    }
    // correction: 0, 1, or 2
    // logarithm: remove 0s -> 1 becomes 0, 2 becomes log(2)
  }
}

/**
 * Needed to convert stanford labels to tokens, as it seems that the CoreLabel are connected to the the full parse tree which
 * then stays in memory.
 * @param index
 * @param word
 */
case class Token(index: Int, word: String, tag: String = "") {
  override def toString: String = word + "-" + index
}

