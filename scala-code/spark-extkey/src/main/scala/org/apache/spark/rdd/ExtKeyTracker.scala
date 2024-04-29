package org.apache.spark.rdd

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.spark.SparkException

import scala.collection.JavaConversions._
import scala.collection.immutable.ListMap
import scala.reflect.ClassTag

class ExtKeyTracker {
  // I can't get this to compile with Scala's mutable map :S
  //val wrappers = new AnyMap[Any]
  val wrappers = new AnyMap
  import ExtKeyTracker.log

  def add[K,V: ClassTag](self: RDD[(ExtKey[K],V)])
              (implicit evk: ClassTag[K]): RDD[(ExtKey[K],V)] = {
    //wrappers += (self, new Wrapper(self,true))
    wrappers.put(self, new Wrapper(self, evk, keys = 1))
    self
  }

  def add[KV1:ClassTag, K2, V2:ClassTag]
    (parent: RDD[KV1], self: RDD[(ExtKey[K2],V2)], keys: Int = 0)
    (implicit evk: ClassTag[K2]): RDD[(ExtKey[K2],V2)] = {
    //if(!wrappers.containsKey(parent))
    //  throw new SparkException("Parent "+parent.hashCode()+" not found")
    val parentWrapper = findWrapper(parent)
    if(parentWrapper.isEmpty) throw new SparkException("Parent "+parent.hashCode()+" not found")
    wrappers.put(self, new Wrapper(self, evk, parentWrapper, keys))
    //wrappers.get(parent)
    self
  }

  def findWrapper(needle: RDD[_], steps: Int = 0): Option[Wrapper] = {
      // Only count as a step if it was a mapper, reduce/shuffle/etc don't add keys (reduce even removes keys)
      // This is kind of a hack...
      val increment = 1
      log.debug("increment: "+increment+"; self: "+needle)
      if (wrappers.containsKey(needle)) {
        log.debug("Found parent in "+steps+" steps")
        Some(wrappers.get(needle))
      } else {
        //needle.dependencies.flatMap((dep: Dependency[_]) => findWrapper(dep.rdd, steps+increment)).flatten.headOption
        val wrappers = for {
          dep <- needle.dependencies
          wrapper <- findWrapper(dep.rdd, steps+increment)
        } yield wrapper
        wrappers.headOption
      }
  }

  def klass(needle: RDD[_]): ClassTag[_] = {
    if(!wrappers.containsKey(needle))
      throw new SparkException("Needle "+needle.hashCode()+" not found")
    wrappers.get(needle).key
  }

  def klasses[T: ClassTag](needle: RDD[T]): ListMap[_ <: RDD[_], ClassTag[_]] = {
    if(!wrappers.containsKey(needle))
      throw new SparkException("Needle "+needle.hashCode()+" not found")
    klasses(wrappers.get(needle))
  }

  /**
   * Iterates over the parents of a wrapper to gather the keys.   *
   * @param wrapper The child wrapper.
   * @param keys Counter to keep track of the key mutations. Only when positive, the key still exists in the rdd used to search with.
   * @return List of key-classes
   * Example:
   * scala> tracker.wrappers.get(groups).keys
   * res24: Int = 1
   * scala> tracker.wrappers.get(groups).parent.get.keys
   * res25: Int = -1
   * scala> tracker.wrappers.get(groups).parent.get.parent.get.keys
   * res26: Int = 0
   * scala> tracker.wrappers.get(groups).parent.get.parent.get.parent.get.keys
   * res27: Int = 1
   * scala> tracker.wrappers.get(groups).parent.get.parent.get.parent.get.parent.get.keys
   * res28: Int = 1
   * scala> tracker.wrappers.get(groups).parent.get.parent.get.parent.get.parent.get.parent.get.keys
   * java.util.NoSuchElementException: None.get
   */
  private def klasses(wrapper: Wrapper, keys: Int = 0): ListMap[_ <: RDD[_], ClassTag[_]] = {
    val key = if (wrapper.keys + keys > 0) ListMap(wrapper.rdd -> wrapper.key) else ListMap.empty
    val count = wrapper.keys + keys + (if (key.nonEmpty) -1 else 0)

    if(wrapper.parent.isDefined)
      key ++ klasses(wrapper.parent.get, count)
    else
      key
  }

  def stepsTo[T1: ClassTag](needle: RDD[T1], self: RDD[_]): Option[Int] = {
    if(!wrappers.containsKey(needle))
      throw new SparkException("Needle "+needle.hashCode()+" not found")
    if(!wrappers.containsKey(self))
      throw new SparkException("Self "+self.hashCode()+" not found")
    stepsTo(wrappers.get(needle), wrappers.get(self), 0)
  }

  private def stepsTo[T1: ClassTag](needle: Wrapper, self: Wrapper, steps: Int): Option[Int] = {
    val increment = self.keys
    //println("stepsTo needle: "+needle.hashCode()+"; self: "+self.hashCode()+"; steps: "+steps+"; increment: "+increment)
    if (self.eq(needle))
      Some(steps)
    else if(self.parent.isDefined)
      stepsTo(needle, self.parent.get, steps+increment)
    else
      None
    //self.dependencies.map((dep: Dependency[_]) => stepsTo(needle, dep.rdd, steps+increment)).flatten.reduceOption(_ min _)
    //http://stackoverflow.com/questions/10922237/scala-min-max-with-optiont-for-possibly-empty-seq
  }

  case class Wrapper
    (rdd: RDD[_], key: ClassTag[_], parent: Option[Wrapper] = Option.empty, keys: Int = 0)

  //class AnyMap[K <: Any] extends mutable.HashMap[K,Wrapper] with mutable.SynchronizedMap[K,Wrapper]

  class AnyMap extends java.util.concurrent.ConcurrentHashMap[Any,Wrapper] {

    // The way this method works changes between versions of Java. Some versions just call get(), which in this case would
    // throw an exception if the key is not found...
    override def containsKey(o: scala.Any): Boolean = {
      super.get(o)!=null
    }

    override def get(key: Any): Wrapper = {
      val res = super.get(key)
      if(res==null)
        throw new SparkException("Wrapper not found for needle "+key.hashCode()+" in haystack "+wrappers.keys().map((v: Any) => v.hashCode()).mkString(","))
      res
    }
  }
}

trait ExtKeys {
  implicit val tracker = new ExtKeyTracker
}

object ExtKeyTracker {
  val log: Log = LogFactory.getLog(this.getClass)
}