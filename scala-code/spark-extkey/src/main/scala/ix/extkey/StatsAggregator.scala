package ix.extkey

import org.apache.commons.math3.stat.descriptive.{StatisticalSummary, AggregateSummaryStatistics, SummaryStatistics}
import java.io.Serializable
import scala.collection.JavaConversions._
import org.apache.spark.util.StatCounter

object StatsAggregator {
  type C = StatCounter

  def create(value: Double): C = {
    //println("creating combiner")
    val stats = new StatCounter()
    stats.merge(value)
    stats
  }

  def add(stats: C, value: Double): C = {
    //println("adding value to combiner")
    stats.merge(value)
    stats
  }

  def merge(one: C, two: C): C = {
    val agg = new StatCounter()
    agg.merge(one)
    agg.merge(two)
    agg
  }

  //stats match {
  //  case summary: SummaryStatistics => summary.addValue(ev.toDouble(value))
  //  case default => throw new Exception("Unknown statistics class "+default)
  //}

  //one match {
  //  case sumOne: SummaryStatistics => two match {
  //    case sumTwo: SummaryStatistics => new AggregateSummaryStatistics(sumOne,sumTwo).getSummary.asInstanceOf[StatisticalSummaryValues]
  //    case default => throw new Exception("Unknown statistics class "+default)
  //  }
  //  case default => throw new Exception("Unknown statistics class "+default)
  //}
}
