package ix.extkey

import scala._
import org.apache.commons.logging.{LogFactory,Log}
import org.apache.spark.rdd.{ExtKey, RDD}
import scala.reflect.ClassTag

trait StatsPreamble {
  private final val log: Log = LogFactory.getLog(this.getClass)

  def rddToStatsFunctions[K,V <: AnyVal]
  (rdd: RDD[(ExtKey[K],V)])
  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]):
  StatsFunctions[K,V] = {
    log.debug("Expanding rdd with statsfunctions")
    new StatsFunctions(rdd)
  }

  // Not very DRY... Nor very intuitive :S Scala's numeric type system :-(
  // Todo: make DRY

  implicit def rddByteToStatsFunctions[K, V <: Byte] (rdd: RDD[(ExtKey[K],V)])
  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]) =
  rddToStatsFunctions(rdd)

  implicit def rddShortToStatsFunctions[K, V <: Short] (rdd: RDD[(ExtKey[K],V)])
  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]) =
  rddToStatsFunctions(rdd)

  implicit def rddIntToStatsFunctions[K, V <: Int] (rdd: RDD[(ExtKey[K],V)])
  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]) =
  rddToStatsFunctions(rdd)

  implicit def rddLongToStatsFunctions[K, V <: Long] (rdd: RDD[(ExtKey[K],V)])
  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]) =
  rddToStatsFunctions(rdd)

  implicit def rddFloatToStatsFunctions[K, V <: Float] (rdd: RDD[(ExtKey[K],V)])
  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]) =
  rddToStatsFunctions(rdd)

  implicit def rddDoubleToStatsFunctions[K, V <: Double] (rdd: RDD[(ExtKey[K],V)])
  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V]) =
  rddToStatsFunctions(rdd)
}