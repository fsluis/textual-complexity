package org.apache.spark.rdd

import ix.util.TimeLogger
import org.apache.commons.logging.{Log, LogFactory}

import scala.reflect.ClassTag

// For future update, consider replacing the list for tuples
// see https://stackoverflow.com/questions/9028459/a-clean-way-to-combine-two-tuples-into-a-new-larger-tuple-in-scala
class ExtKey[K: ClassTag] protected (val key: K, val history: List[Any] = List())
  extends Tuple2[K,List[Any]](key,history) {

  override def toString() =
    //key.toString+"'"
    "(" + concat().reverse.mkString(",") + ")"

  def concat() = {
    key :: history
  }

  /**
   * Redirects to the hashCode implementation of the (most recent) key,
   * to assure compatibility with Spark methods that only use the most recent key.
   * @return key.hashCode
   */
  /*override def hashCode(): Int = {
    key.hashCode()
  } */
  override def hashCode(): Int = concat.reverse.hashCode

  override def equals(x$1: Any): Boolean = ExtKey.tlog.time { super.equals(x$1) }

  /**
   * Redirects to the equals implementation of the (most recent) key,
   * to assure compatibility with Spark methods that only use the most recent key.
   * @return key.equals
   */
  //ToDo: make this a case class and replace '._1' for '.key' ?
  /*override def equals(other: Any): Boolean = other match {
      case ext: ExtKey[K] => key.equals(ext._1)
      case _ => super.equals(other)
  } */
  // Probably the next function is already implemented (better) by Scala's Tuple2
  /*override def equals(other: Any): Boolean = other match {
      case ext: ExtKey[K] => if(!key.equals(ext._1)) false else history.equals(ext._2)
      case _ => super.equals(other)
  }*/

  // kl k4 k3 k2 k1 k0
  // k0 k1 k2 k3 k4 kl
  // positive: drop the most recent n keys
  // negative: drop evt but the most recent n keys (take the most recent n keys)

  def take(n: Int): ExtKey[K] = {
    if(n>0)
      new ExtKey[K](key, history.take(n-1))
    else //nothing to be done here, fail-safe func this
      this
  }

  def drop[K2: ClassTag](n: Int): ExtKey[K2] = {
    if(n>0) {
      val dropped = history.drop(n-1)
      new ExtKey(dropped.head.asInstanceOf[K2], dropped.tail)
    } else //nothing to be done here, fail-safe func this
      this.asInstanceOf[ExtKey[K2]]
  }
  
  def getKey[T](n: Int = history.size): T = {
    val res = concat().reverse.apply(n)
    res.asInstanceOf[T]
  }
  
  def head[T] = getKey[T](0)
  def tail = key
}

object ExtKey {
  private final val log: Log = LogFactory.getLog(this.getClass)
  private final val tlog: TimeLogger = new TimeLogger(log, "extkey.equals")

  def apply[K:ClassTag](key: K): ExtKey[K] =
    new ExtKey(key)

  //def key[K:ClassTag](key: K, ext: ExtKey[K]): ExtKey[K] =
  //  new ExtKey(key, ext.concat)

  def apply[K:ClassTag, T:ClassTag](key: K, ext: ExtKey[T]): ExtKey[K] =
    new ExtKey(key, ext.concat)

  def apply[K1:ClassTag,K2:ClassTag](one: K1, two: K2): ExtKey[K2] =
    apply(two, apply(one))

  //def apply[T, K:ClassTag](key: K, ext: ExtKey[T])(implicit m: Manifest[T]): ExtKey[K] = {
  //  new ExtKey(key, ext)
  //}
}