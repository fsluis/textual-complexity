package ix.complexity.wordnet

import java.util

import edu.mit.jwi.item.{ISynsetID, POS}
import ix.common.util.TimeLogger
import ix.complexity.wordnet.model.{Node, StOngeNode, WordNetUtil2}
import ix.util.services2.{ModelService, Models}
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable

object WordNetService extends ModelService[WordNetUtil2] {

  val log: Log = LogFactory.getLog(this.getClass)
  val tlog = new TimeLogger(log, "wordnet", 10000)
  val ALL_POS = List(POS.ADJECTIVE,POS.ADVERB,POS.NOUN,POS.VERB)

  def load(name: String, confOpt: Option[Configuration]): WordNetUtil2 = {
    val dir = confOpt.get.getString("wordnet-dir")
    new WordNetUtil2(dir)
  }

  def findNode[N <: Node](word: String, poses: Seq[POS] = WordNetService.ALL_POS, nodeType: Class[N], modelName: String = Models.DEFAULT): Node = {
    val t = tlog.start()
    val model = get(modelName)
    val synsets = model.findSynsets(word, poses)
    val node = nodeType.newInstance
    node.addSynsets(synsets)
    tlog.stop(t)
    node
  }

  /**
   *
   * @param node Node to calculate spread for
   * @param maxDepth Number of steps from starting synset(s) to find related nodes
   * @param modelName WordNet model
   * @tparam N NodeType to use (specifies which related nodes to include)
   * @return cumulative spread, does not contain null
   */
  def computerSpread[N <: Node](node: N, maxDepth: Int = 5, modelName: String = Models.DEFAULT): (Array[Integer], Array[Integer]) = {
    val model = get(modelName)
    val spreads = WordNetSpread.doMap(node, maxDepth, model)
    val spreadCumulative = spreads(1)
    // small fix for null values in later positions at cumulative spread
    // as these values are cumulative null values should not follow on
    var prev = spreadCumulative(0)
    for(i <- 1 until spreadCumulative.size)
      if (spreadCumulative(i)==null)
        spreadCumulative(i) = prev
    spreadCumulative
  }

}
