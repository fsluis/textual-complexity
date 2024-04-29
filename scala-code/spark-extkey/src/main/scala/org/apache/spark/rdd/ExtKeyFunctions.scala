package org.apache.spark.rdd

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.spark.rdd.ExtKeyFunctions._

import scala.reflect.ClassTag


/**
 * Extra functions available on RDDs of (key, value) pairs through an implicit conversion.
 * Import `org.apache.spark.SparkContext._` at the top of your program to use these functions.
 * Types: Extended key, Key, History, Value, Product
 */
class ExtKeyFunctions[
  K: ClassTag,
  V: ClassTag] (
  self: RDD[(ExtKey[K],V)])
  extends PairRDDFunctions[ExtKey[K],V](self) {

  def test() = {
    println("Test! :-)")
  }

  /**
   * Drop the most recent n keys, possibly until <i>that</i>.
   * Assumes that each rdd in between this rdd and that rdd also produced a new key! If not, the count of the number
   * of keys is incorrect and undetermined behavior follows.
   * Throws an exception if the rdd (that) was not found in the dependencies of this rdd.
   * @param that rdd of the new most recent key
   * @param n number of keys to drop. If < 0 then the keys will be dropped until <i>that</i>
   * @return mapped rdd with new key K2 and values V
   */
  //def dropKey[K2: ClassTag, V2: ClassTag](that: RDD[(ExtKey[K2],V2)], n: Int = -1): RDD[(ExtKey[K2],V)] = {
  def dropKey[K2,V2](that: RDD[(ExtKey[K2],V2)], n: Int = -1)(implicit evk: ClassTag[K2], evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K2],V)] = {
    val m = if (n>=0) n else tracker.stepsTo(that,self).get //throws an exception if rdd is not found
    log.info("dropKey: Dropping "+m+" levels back to key "+that)
    val f: ((ExtKey[K],V)) => (ExtKey[K2],V) =
      kv => (kv._1.drop(m), kv._2)
    tracker.add(self, self.map(f), keys = -1*m)
    //self.map(kv => (kv._1.drop(n), kv._2))
  }

  //def groupByKey[K2: ClassTag, V2: ClassTag](that: RDD[(ExtKey[K2],V2)]): RDD[(ExtKey[K2], Seq[V])] = {
  def groupByKey[K2,V2](that: RDD[(ExtKey[K2],V2)])(implicit evk: ClassTag[K2], evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K2], Iterable[V])] = {
    val dropped = dropKey(that)
    tracker.add(dropped, dropped.groupByKey(), keys = 0)
  }

  /**
   * Keep only (take) the most recent n keys
   * @param n number of keys to drop
   * @return mapped rdd with key K and values V
   */
  def takeKey(n: Int): RDD[(ExtKey[K], V)] = {
    log.info("Taking the last "+n+" levels of keys")
    //val f: ((E,V)) => (ExtKey[K],V) = kv => (kv._1.take(n), kv._2)
    //self.map(f)
    self.map(kv => (kv._1.take(n), kv._2))
  }

  /**
   * Keep only (take) the most recent n keys until and including that
   * Assumes that each rdd in between this rdd and that rdd also produced a new key! If not, the count of the number
   * of keys is incorrect and undetermined behavior follows.
   * Throws an exception if the rdd (that) was not found in the dependencies of this rdd.
   * @param that The rdd until and including which to take the keys.
   * @return A new RDD with taken keys.
   */
  //def takeKey[K2: ClassTag, V2: ClassTag](that : RDD[(ExtKey[K2],V2)]): RDD[(ExtKey[K],V)] = {
  def takeKey[K2,V2](that : RDD[(ExtKey[K2],V2)])
                    (implicit evk: ClassTag[K2], evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K],V)] = {
    val n = tracker.stepsTo(that, self).get //throws an exception if rdd is not found
    val res = takeKey(n+1) //inclusive!
    tracker.add(self, res, keys = 0)
  }

  //def extFlatMap[K2: ClassTag,V2: ClassTag](f: ((K,V)) => TraversableOnce[(K2,V2)]): RDD[(ExtKey[K2],V2)] = {
  def kvFlatMap[K2,V2](f: ((K,V)) => TraversableOnce[(K2,V2)])
                      (implicit evk: ClassTag[K2], evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K2],V2)] = {
    log.debug("Expanded flatMap")
    // g wraps f (an iterator) as to convert its input from ExtKey[K] to K and its output from K2 to ExtKey[K2]
    val g: ((ExtKey[K],V)) => TraversableOnce[(ExtKey[K2],V2)] = kv => {
      new Iterator[(ExtKey[K2],V2)] {
        val traversable: Iterator[(K2,V2)] = f.apply((kv._1.key, kv._2)).toIterator
        def hasNext: Boolean = traversable.hasNext
        def next(): (ExtKey[K2], V2) = {
          val kv2 = traversable.next()
          (ExtKey(kv2._1, kv._1), kv2._2)
        }
      }
    }
    tracker.add(self, self.flatMap(g), keys = 1)
  }

  //def eflatMap[V2](f: ((K,V)) => TraversableOnce[V2])(implicit evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[Int],V2)] = {
  def vFlatMap[V2](f: ((K,V)) => Iterator[V2])(implicit evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[Int],V2)] = {
    this.kvFlatMap(
      (x: (K, V)) => f(x).zipWithIndex.map(_.swap).toTraversable
    )
  }

  def optMap[V2](f: (V) => Option[V2])(implicit evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K],V2)] = {
    log.debug("Expanded flatMap")
    // g wraps f (an iterator) as to convert its input from ExtKey[K] to K and its output from K2 to ExtKey[K2]
    val g: ((ExtKey[K],V)) => TraversableOnce[(ExtKey[K],V2)] = kv => {
        val value = f.apply( kv._2 )
        if(value.isDefined)
          Some( (kv._1, value.get) )
        else
          None
      }
    tracker.add(self, self.flatMap(g), keys = 0)
  }

  // implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]):
  // def extMap[K2: ClassTag,V2: ClassTag](f: ((K,V)) => (K2,V2)): RDD[(ExtKey[K2],V2)] = {
  def kvMap[K2,V2](f: ((K,V)) => (K2,V2))(implicit evk: ClassTag[K2], evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K2],V2)] = {
    log.debug("Expanded map")
    // g wraps f as to convert its input from ExtKey[K] to K and its output from K2 to ExtKey[K2]
    val g: ((ExtKey[K],V)) => (ExtKey[K2],V2) = kv => {
      val res = f.apply((kv._1.key, kv._2))
      (ExtKey(res._1, kv._1), res._2)
    }
    tracker.add(self, self.map(g), keys = 1)
  }

  /**
   * A map operation which doesn't add/change the key
   */
  def vMap[V2](f: (V) => V2)(implicit evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K],V2)] = {
    log.debug("Expanded map")
    // g wraps f as to convert its input from ExtKey[K] to K and its output from K2 to ExtKey[K2]
    val g: ((ExtKey[K],V)) => (ExtKey[K],V2) = kv => {
      (kv._1, f.apply(kv._2))
    }
    tracker.add(self, self.map(g), keys = 0)
  }

  /**
   * A map operation which doesn't add/change the key
   */
  def kvvMap[V2](f: (K,V) => V2)(implicit evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K],V2)] = {
    log.debug("Expanded map")
    // g wraps f as to convert its input from ExtKey[K] to K and its output from K2 to ExtKey[K2]
    val g: ((ExtKey[K],V)) => (ExtKey[K],V2) = kv => {
      (kv._1, f.apply(kv._1.key, kv._2))
    }
    tracker.add(self, self.map(g), keys = 0)
  }

  /**
   * A map operation which adds a key, w/o knowledge of the previous key
   */
  def valueToKeyValue[K2,V2](f: (V) => (K2,V2))(implicit evk: ClassTag[K2], evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K2],V2)] = {
    log.debug("Expanded map")
    // g wraps f as to convert its input from ExtKey[K] to K and its output from K2 to ExtKey[K2]
    val g: ((ExtKey[K],V)) => (ExtKey[K2],V2) = kv => {
      val res = f.apply(kv._2)
      (ExtKey(res._1, kv._1), res._2)
    }
    tracker.add(self, self.map(g), keys = 1)
  }

  /**
   * A map operation which adds a key, w/o knowledge of the previous key
   */
  def valueToKey[K2](f: (V) => K2)(implicit evk: ClassTag[K2], tracker: ExtKeyTracker): RDD[(ExtKey[K2], V)] = {
    log.debug("Expanded map")
    // g wraps f as to convert its input from ExtKey[K] to K and its output from K2 to ExtKey[K2]
    val g: ((ExtKey[K],V)) => (ExtKey[K2],V) = kv => {
      val res = f.apply(kv._2)
      (ExtKey(res, kv._1), kv._2)
    }
    tracker.add(self, self.map(g), keys = 1)
  }

  /**
   * Helper function that moves keys to values and subsequently drops the keys.
   */
  def disentangle[K2,V2,K3,V3,K4,V4,K5,V5]
  (first: RDD[(ExtKey[K2],V2)], second: RDD[(ExtKey[K3],V3)], third: RDD[(ExtKey[K4],V4)], fourth: RDD[(ExtKey[K5],V5)])
  (implicit evk2: ClassTag[K2], evv2: ClassTag[V2], evk3: ClassTag[K3], evv3: ClassTag[V3], evk4: ClassTag[K4], evv4: ClassTag[V4], evk5: ClassTag[K5], evv5: ClassTag[V5], tracker: ExtKeyTracker):
  RDD[(ExtKey[K5], (K,K2,K3,K4,V))] = {
    this.
      kvvMap((k,v) => (k,v)).dropKey(first).
      kvvMap((k2: K2, t: (K, V)) => (t._1, k2, t._2) ).dropKey(second).
      kvvMap((k3: K3, t: (K, K2, V)) => (t._1, t._2, k3, t._3)).dropKey(third).
      kvvMap((k4: K4, t: (K, K2, K3, V)) => (t._1, t._2, t._3, k4, t._4)).dropKey(fourth)
  }

  /**
   * Helper function that moves keys to values and subsequently drops the keys.
   */
  def disentangle[K2,V2,K3,V3,K4,V4]
    (first: RDD[(ExtKey[K2],V2)], second: RDD[(ExtKey[K3],V3)], third: RDD[(ExtKey[K4],V4)])
    (implicit evk2: ClassTag[K2], evv2: ClassTag[V2], evk3: ClassTag[K3], evv3: ClassTag[V3], evk4: ClassTag[K4], evv4: ClassTag[V4], tracker: ExtKeyTracker):
    RDD[(ExtKey[K4], (K,K2,K3,V))] = {
    this.
      kvvMap((k,v) => (k,v)).dropKey(first).
      kvvMap((k2: K2, t: (K, V)) => (t._1, k2, t._2) ).dropKey(second).
      kvvMap((k3: K3, t: (K, K2, V)) => (t._1, t._2, k3, t._3)).dropKey(third)
  }

  /**
   * Helper function that moves keys to values and subsequently drops the keys.
   */
  def disentangle[K2,V2,K3,V3]
  (first: RDD[(ExtKey[K2],V2)], second: RDD[(ExtKey[K3],V3)])
  (implicit evk2: ClassTag[K2], evv2: ClassTag[V2], evk3: ClassTag[K3], evv3: ClassTag[V3], tracker: ExtKeyTracker):
  RDD[(ExtKey[K3], (K,K2,V))] = {
    this.
      kvvMap((k,v) => (k,v)).dropKey(first).
      kvvMap((k2: K2, t: (K, V)) => (t._1, k2, t._2) ).dropKey(second)
  }

  /**
   * Helper function that moves keys to values and subsequently drops the keys.
   */
  def disentangle[K2,V2]
  (first: RDD[(ExtKey[K2],V2)])
  (implicit evk2: ClassTag[K2], evv2: ClassTag[V2], tracker: ExtKeyTracker):
  RDD[(ExtKey[K2], (K,V))] = {
    this.
      kvvMap((k,v) => (k,v)).dropKey(first)
  }

  /**
   * Helper function that moves values to keys.
   */
  def entangle4[K2,K3,K4,K5]
  (f: (V) => (K2,K3,K4,K5))
  (implicit evk2: ClassTag[K2], evk3: ClassTag[K3], evk4: ClassTag[K4], evk5: ClassTag[K5], tracker: ExtKeyTracker):
  RDD[(ExtKey[K5], V)] = {
    this.
      valueToKey(f(_)._1).
      valueToKey(f(_)._2).
      valueToKey(f(_)._3).
      valueToKey(f(_)._4)
  }

  /**
   * Helper function that moves values to keys.
   */
  def entangle3[K2,K3,K4]
  (f: (V) => (K2,K3,K4))
  (implicit evk2: ClassTag[K2], evk3: ClassTag[K3], evk4: ClassTag[K4], tracker: ExtKeyTracker):
  RDD[(ExtKey[K4], V)] = {
    this.
      valueToKey(f(_)._1).
      valueToKey(f(_)._2).
      valueToKey(f(_)._3)
  }

  /**
   * Helper function that moves values to keys.
   */
  def entangle2[K2,K3]
  (f: (V) => (K2,K3))
  (implicit evk2: ClassTag[K2], evk3: ClassTag[K3], tracker: ExtKeyTracker):
  RDD[(ExtKey[K3], V)] = {
    this.
      valueToKey(f(_)._1).
      valueToKey(f(_)._2)
  }

  /**
   * Helper function that moves values to keys.
   */
  def entangle1[K2]
  (f: (V) => K2)
  (implicit evk2: ClassTag[K2], tracker: ExtKeyTracker):
  RDD[(ExtKey[K2], V)] = {
    this.
      valueToKey(f(_))
  }

  def valueFilter(f: (V) => Boolean)(implicit tracker: ExtKeyTracker): RDD[(ExtKey[K],V)] = {
    log.debug("Value filter")
    tracker.add(self, self.filter( kv => f(kv._2) ), keys = 0)
  }

  def keyFilter(f: (K) => Boolean)(implicit tracker: ExtKeyTracker): RDD[(ExtKey[K],V)] = {
    log.debug("Key filter")
    tracker.add(self, self.filter( kv => f(kv._1.key) ), keys = 0)
  }

  def reduceByKey[K2,V2](that: RDD[(ExtKey[K2],V2)], f: (V,V) => V)
                        (implicit evk: ClassTag[K2], evv: ClassTag[V2], tracker: ExtKeyTracker): RDD[(ExtKey[K2], V)] = {
    val dropped = dropKey(that)
    tracker.add(dropped, dropped.reduceByKey(f), keys = 0)
  }

  //def stepsTo[K2: ClassTag, V2: ClassTag](needle: RDD[(ExtKey[K2],V2)]): Option[Int] =
  //  ExtKeyFunctions.stepsTo(needle, self)
  /**
   * Wrapper for the normal reduceByKey function
   */
  def extReduceByKey(func: (V, V) => V)
    (implicit tracker: ExtKeyTracker): RDD[(ExtKey[K], V)] =
    tracker.add(self, super.reduceByKey(func), keys=0)

  /**
   * Wrapper for the normal reduceByKey function
   * createCombiner: V => C,
      mergeValue: (C, V) => C,
      mergeCombiners: (C, C) => C,
      numPartitions: Int
   */
  def extCombineByKey[C](createCombiner: V => C, mergeValue: (C,V) => C, mergeCombiners: (C,C) => C, numPartitions: Int)
                    (implicit evk: ClassTag[C], tracker: ExtKeyTracker): RDD[(ExtKey[K], C)] =
    tracker.add(self, super.combineByKey(createCombiner, mergeValue, mergeCombiners, numPartitions), keys=0)

  def aggregateByKey[U,K2,V2](that: RDD[(ExtKey[K2],V2)], zeroValue: U)(seqOp: (U,V) => U, combOp: (U,U) => U)
                          (implicit evk: ClassTag[K2], evv: ClassTag[V2], evu: ClassTag[U], tracker: ExtKeyTracker): RDD[(ExtKey[K2], U)] = {
    val dropped = dropKey(that)
    tracker.add(dropped, dropped.aggregateByKey(zeroValue)(seqOp, combOp))
  }

  def extAggregateByKey[U](zeroValue: U)(seqOp: (U,V) => U, combOp: (U,U) => U)
                          (implicit evk: ClassTag[U], tracker: ExtKeyTracker): RDD[(ExtKey[K], U)] = {
    tracker.add(self, super.aggregateByKey(zeroValue)(seqOp, combOp))
  }

  /**
   * Wrapper for union that keeps track of changes
   * @param that
   * @param tracker
   * @return
   */
  def extUnion(that: RDD[(ExtKey[K],V)])
              (implicit tracker: ExtKeyTracker): RDD[(ExtKey[K], V)] = {
    tracker.add(self, self.union(that))
  }
}

object ExtKeyFunctions extends ExtKeyPreamble {
  private final val log: Log = LogFactory.getLog(this.getClass)

  // ToDo: find a strict solution to this! (i.e., one not based on assumption or reverse engineering)
  /*private def stepsTo[T1: ClassTag](needle: RDD[T1], self: RDD[_], steps: Int = 0): Option[Int] = {
    // Only count as a step if it was a mapper, reduce/shuffle/etc don't add keys (reduce even removes keys)
    // This is kind of a hack...
    val increment = if (self.isInstanceOf[MappedRDD[_,_]] || self.isInstanceOf[FlatMappedRDD[_,_]]) 1 else 0
    log.debug("increment: "+increment+"; self: "+self)
    if (self.eq(needle))
      Some(steps)
    else
      self.dependencies.map((dep: Dependency[_]) => stepsTo(needle, dep.rdd, steps+increment)).flatten.reduceOption(_ min _)
    //http://stackoverflow.com/questions/10922237/scala-min-max-with-optiont-for-possibly-empty-seq
  } */
}