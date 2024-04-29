package ix.complexity.lucene3.esa

import gnu.trove.map.hash.TIntDoubleHashMap
import ix.complexity.features.word.EsaFeatures
import ix.complexity.lucene3.tokenizer.Tokenizer
import ix.util.services2.{ModelService, Models, ServiceException}
import org.apache.commons.configuration.Configuration

object EsaService extends ModelService[EsaModelWrapper] {

  def load(name: String, confOpt: Option[Configuration]): EsaModelWrapper = {
    val conf = confOpt.get
    if (!conf.containsKey("index")) throw new ServiceException("No index specified for esa model " + name)
    val threshold: Double = conf.getDouble("weighting_threshold", EsaModel.WEIGHTING_THRESHOLD)
    val cache: Int = conf.getInt("cache_size", EsaModel.CACHE_SIZE)
    val index: String = conf.getString("index")
    val stemmer: String = conf.getString("stemmer")
    new EsaModelWrapper(index, threshold, cache, stemmer)
  }

  /**
   * Turn a token (word) into a topic vector.
   * @param token A stemmed word. The stemmer to be used depends on the training of the ESA model.
   * @param model The ESA model to be used. Refers to the indexed models in the Services definitions.
   * @return
   */
  def topic(token: String, model: String = Models.DEFAULT): Option[TIntDoubleHashMap] = {
    Option(get(model).getConceptMap(token))
  }

  def topics(sentence: String, model: String = Models.DEFAULT, baseOffset:Long = 0): Iterator[(Long,Option[TIntDoubleHashMap])] = {
    val m = get(model)
    val tokens = Tokenizer.tokenize(sentence, m.stemmer, baseOffset)
    tokens.map{case (pos, token) => pos -> Option(m.getConceptMap(token))}
  }

  def sentence(sentence: String, model: String = Models.DEFAULT) = {
    val vectors = topics(sentence,model).map(_._2).filter(_.isDefined).map(_.get)
    EsaFeatures.centroid(vectors)
  }

}

class EsaModelWrapper(index: String, weightingThreshold: Double, cacheSize: Int, val stemmer: String = "")
  extends EsaModel(index, weightingThreshold, cacheSize)