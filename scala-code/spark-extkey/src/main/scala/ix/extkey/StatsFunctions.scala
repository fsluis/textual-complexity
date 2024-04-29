package ix.extkey

import org.apache.spark.rdd._

import scala._
import org.apache.commons.logging.LogFactory
import org.apache.commons.logging.Log

object StatsFunctions extends StatsPreamble with ExtKeyPreamble {
  private final val log: Log = LogFactory.getLog(this.getClass)

  //private def mergeTwo[K,V1,V2](self: RDD[(ExtKey[K],V1)], that: RDD[(ExtKey[K],V2)]) = {
    //self.leftOuterJoin(that).map((tuple: (ExtKey[K], (V1, Option[V2]))) => )
    //self.cogroup(that).map((tuple: (ExtKey[K], (Seq[V1], Seq[V2]))) => )
    //self.join()
  //}

  //def merge[K,V <: AnyVal](rdds: RDD[(ExtKey[K],V)]*) = {
    //rdds.reduce(mergeTwo(_,_))
  //}

  //def mergeA[K,V <: AnyVal]( stats: List[String], rdds: (String,RDD[(ExtKey[K],V)])* ) = mergeB(stats, Map(rdds))

  /*def mergeB[K,V <: AnyVal]( stats: List[String], features: Map[String,RDD[(ExtKey[K],V)]] ) = {
    //val rdds = features.map{ case(key,rdd) => (key, if(key.endsWith("*")) rddToStatsFunctions(rdd).statsPerKey() else rdd ) }
    val x = statz(features.head._2)
    //val rdds = features.map((t: (String, RDD[(ExtKey[K], V)])) => (t._1, if(t._1.endsWith("*")) rddToStatsFunctions(t._2).statsPerKey() else t._2 ))
    val names = for(key <- features.keys)
      yield if(key.endsWith("*"))
        product(key.stripSuffix("*"), stats)
      else
        List(key)

  }

  // This gives an error :S Don't understand it...
  def statz[K,V <: AnyVal](rdd: RDD[(ExtKey[K],V)]) = {
    //import StatsFunctions.rddToStatsFunctions
    rdd.statsPerKey
  }

  // grouper, rdds[k,anyval],
  // implicit conversion? from k to grouper??? (cannot be done I think, diff k's)
  // at least not necessary yet
  // name -> rdd[extkey[k],anyval]*
  // features: (String,RDD)*
  // stats in name? -> stats, expand name (sub-sequences?)
  // res: rdd[K,Seq[AnyVal]], names
  // merge rdds, add value per feature/derivative
  // how to guarantee sequence?

  // alternative: words.join("spw" -> rdd).join("cpw*" -> rdd)
  //
    */

  def product(names: List[String], appendices: List[String], originals: Boolean = true): List[String] = {
    val res1 = if(originals) names else List()
    val res2 = for(name <- names)
      yield product(name,appendices)
    res1 ::: res2.flatten
  }

  private def product(name: String, appendices: List[String]): List[String] = {
    for(appendix <- appendices)
      yield name+appendix
  }

  /*private def mergeTwo[K,V1,V2](self: RDD[(ExtKey[K],V1)], that: RDD[(ExtKey[K],V2)]): RDD[(ExtKey[K],(Iterable[V1],Iterable[V2]))] = {
    rddToPairRDDFunctions(self).
      cogroup(that).
      map((t: (ExtKey[K], (Iterable[F], Iterable[F]))) =>
      (t._1, (t._2._1 ++ t._2._2).flatten.toMap) )
  }

  def merge[K](rdds: RDD[(ExtKey[K],F)]*) = {
    rdds.reduce(mergeTwo(_,_))
  }*/

}

class StatsFunctions[K,V <: AnyVal] (
  self: RDD[(ExtKey[K],V)])
  (implicit evk: ClassManifest[K], evv: ClassManifest[V], evn: Numeric[V])
  extends ExtKeyFunctions[K,V](self) {

  import StatsAggregator._
  import StatsFunctions._

  def statsPerKey(implicit tracker: ExtKeyTracker): RDD[(ExtKey[K],C)] = {
    log.debug("stats per key, calling combinebykey")
    tracker.add(self,
      combineByKey[C](
      (v: V) => StatsAggregator.create(evn.toDouble(v)),
      (c: C, v: V) => StatsAggregator.add(c,evn.toDouble(v)),
      (one: C, two: C) => StatsAggregator.merge(one, two))
    , 0)
  }

  def stats[K2,V2](that: RDD[(ExtKey[K2],V2)])
    (implicit evk2: ClassManifest[K2], evv2: ClassManifest[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K2], C)] =
    rddToStatsFunctions(dropKey(that)).statsPerKey

}
