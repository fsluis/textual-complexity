package ix.util

/**
 * Created by f on 09/07/16.
 */
object ScalaHelper {
  implicit def bool2int(b:Boolean): Int = if (b) 1 else 0

  implicit def map2Properties(map: Map[String,String]):java.util.Properties =
    (new java.util.Properties /: map) {case (props, (k,v)) => props.put(k,v); props}

  /**
   * Splits a list into a list-of-list, combining consecutive identical elements into sublists whilst preserving order
   * From https://stackoverflow.com/questions/4761386/scala-list-function-for-grouping-consecutive-identical-elements
   * @param list
   * @tparam T
   * @return
   */
  def split[T](list: List[T]) : List[List[T]] = list match {
    case Nil => Nil
    case h::t => val segment = list takeWhile {h ==}
      segment :: split(list drop segment.length)
  }
}
