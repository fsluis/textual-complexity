package ix.complexity.features.word

import ix.complexity.lucene3.esa.{CosineKernel, WindowedCentroid, EsaUtil, EsaService}
import ix.util.services2.Models
import gnu.trove.map.hash.TIntDoubleHashMap
import scala.collection.JavaConversions._
import ix.complexity.features.common.ReferenceFeatures

trait EsaUtil {

  //def topics(tokens: Iterable[String], model:String = Models.DEFAULT): Iterator[TIntDoubleHashMap] = topics(tokens.iterator,model)
  def topics(tokens: Iterator[String], model:String = Models.DEFAULT): Iterator[TIntDoubleHashMap] = {
    val esa = EsaService.get(model)
    tokens.map((t: String) => esa.getConceptMap(t))
  }

  def centroid(topics: Iterable[TIntDoubleHashMap]): TIntDoubleHashMap = EsaUtil.average(topics.iterator)
  def centroid(topics: Iterator[TIntDoubleHashMap]): TIntDoubleHashMap = EsaUtil.average(topics)

  def windowedSums(topics: Iterable[TIntDoubleHashMap], window: Int): Iterator[(Int,TIntDoubleHashMap)] = {
    val centroids = new WindowedCentroid(topics.iterator, window, false)
    centroids.zipWithIndex.map((t: (TIntDoubleHashMap, Int)) => (t._2,t._1) )
  }
  def centroids(topics: Iterable[TIntDoubleHashMap], window: Int): Iterator[(Int,TIntDoubleHashMap)] = centroids(topics.iterator, window)
  def centroids(topics: Iterator[TIntDoubleHashMap], window: Int): Iterator[(Int,TIntDoubleHashMap)] = {
    val centroids = new WindowedCentroid(topics, window, true)
    centroids.zipWithIndex.map((t: (TIntDoubleHashMap, Int)) => (t._2,t._1) )
  }

  def entropy(topic: TIntDoubleHashMap) = EsaUtil.entropy(topic)

  def cohesion(topics: Iterable[TIntDoubleHashMap], maxK: Int = 5, weigh: Boolean = true) = {
    val kernel = new CosineKernel
    val matrix = kernel.matrix(asJavaIterable(topics))
    val features = new ReferenceFeatures
    features.setMaxK(maxK)
    features.setWeigh(weigh)
    features.cohesion(matrix)
  }
}

object EsaFeatures extends EsaUtil