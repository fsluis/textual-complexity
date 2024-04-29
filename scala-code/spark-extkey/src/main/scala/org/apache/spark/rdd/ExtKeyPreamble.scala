package org.apache.spark.rdd

import org.apache.commons.logging.{Log, LogFactory}
import scala.reflect.ClassTag


trait ExtKeyPreamble {
  private final val log: Log = LogFactory.getLog(this.getClass)

  implicit def rddToExtKeyFunctions[K: ClassTag, V: ClassTag](rdd: RDD[(ExtKey[K], V)]): ExtKeyFunctions[K,V] = {
    log.debug("Expanding rdd with extkeyfunctions")
    new ExtKeyFunctions(rdd)
  }

  //implicit def rddToExtKeyFunctionsViaMapper[K: ClassTag, V: ClassTag](rdd: RDD[(K, V)]): ExtKeyFunctions[K,V] = {
  //  println("Expanding rdd with extkeyfunctions")
  //  new ExtKeyFunctions(rddToExtKeyRDD(rdd))
  //}

  def ext[K,V](rdd: RDD[(K, V)])(implicit evk: ClassTag[K], evv: ClassTag[V], tracker: ExtKeyTracker): RDD[(ExtKey[K],V)] = {
    log.debug("Converting rdd to extkeyrdd via map operation")
    val f: ((K,V)) => (ExtKey[K],V) =
      kv => (ExtKey(kv._1), kv._2)
    tracker.add(rdd.map(f))
  }
}