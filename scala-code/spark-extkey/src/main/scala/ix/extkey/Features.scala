package ix.extkey

import org.apache.commons.math3.stat.descriptive.StatisticalSummary
import org.apache.spark.Partitioner._
import org.apache.spark.SparkException
import org.apache.spark.rdd._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, _}

import scala.Predef._
import scala.collection.mutable
import scala.reflect.ClassTag

//import ExtKeyFunctions._
import ix.extkey.StatsFunctions._

// name -> (rdd, num, structtype, stats [string/null], statsRdd)
case class Feature( rdd: RDD[_ <: Product2[ExtKey[_], AnyVal]],
                    num: Numeric[_ <: AnyVal],
                    struct: DataType,
                    stats: Option[List[String]],
                    groupedRDD: Option[RDD[_ <: Product2[ExtKey[_], StatsAggregator.C]]] = Option.empty)

@transient
class Features {
  @transient val features = new mutable.LinkedHashMap[String,  Feature]
  @transient val keys = new mutable.LinkedHashMap[RDD[_], String]

  def dataFrame[K,V](grouper: RDD[(ExtKey[K],V)], parameters: List[String] = List("mean"))
              (implicit evk: ClassTag[K], evv: ClassTag[V], tracker: ExtKeyTracker) = {
    val extraKeys = this.keys.clone()
    //val extraKeys = features.keys.clone()
    // Third, create the schema
    val keys = tracker.klasses(grouper).zipWithIndex.map((t: ((RDD[_], ClassTag[_]), Int)) => {
      val ((rdd, klass), i) = t
      val name = extraKeys.remove(rdd).getOrElse("unknown_key_"+i)
      rdd.setName(name)
        //this.keys.getOrElse(rdd, "unknown_key_"+i)
      //name
      StructField(name, getType(klass), nullable = true)
    })

    // Add the keys that were not found in the dataset as dummmy/empty columns
    val extra = extraKeys.map{case (rdd: RDD[_], name: String) =>
      StructField(name, getType(tracker.klass(rdd)), nullable = true)
      //10 -> name
    }

    // First, group rdds (drop keys) on that
    val grouped = group(grouper, parameters)

    // Second, combine the rdds into Row objects
    // Calling toSeq here gives serialization error. Sigh...
    val g_rdds = grouped.values.toList
    //val m_rdds = g_rdds.map(_.vMap(_.mean))
    //val left = m_rdds.head.vMap(Seq(_))
    //val rows = features.recursive_join(left, m_rdds.drop(1))
    val rdd = new CoGroupedRDD[ExtKey[_]](g_rdds, defaultPartitioner(g_rdds.head, g_rdds.drop(1):_*))
    val rows = rdd.map( Features.toRow(_, extra.size, parameters) )
    // todo: the sequence of stats will differ between data gathering and grouping
    // -> make parameters a sequence of index -> list[String]

    // make sure to flatten the keys after transforming them to a list,
    // otherwise the sequence changes!
    val columns = keys.toList.reverse ::: extra.toList ::: grouped.keys.toList.flatten
    val schema = StructType(columns)

    //this.keys: rdd -> string
    //tracker.klasses: rdd -> classtag V
    //iterate over the tracker.klasses V
    //if there's a name for an rdd, then use it V
    //otherwise give a temp name: "unknown_key_0" V
    // + make this.features serializable to prevent these idiotic errosr

    //val types = rdds.map((rdd: RDD[_ <: Product2[ExtKey[_], _]]) => Features.getType(rdd))
    (rows, schema, columns, grouped)
    //sql.createDataFrame(rows, schema)
  }
  def recursive_join[K,V](left: RDD[(ExtKey[K], Seq[V])], rdds: List[RDD[(ExtKey[K], V)]])
                       (implicit evk: ClassTag[K], evv: ClassTag[V], tracker: ExtKeyTracker): RDD[(ExtKey[K], Seq[V])] = {
    if(rdds.isEmpty) return left
    val right = rdds.head
    val newLeft = left.join(right).vMap{ case (seqV,v) => seqV :+ v }
    val newRight = rdds.drop(1)
    recursive_join(newLeft, newRight)
  }

  def group[K,V](that: RDD[(ExtKey[K],V)], selection: List[String])
                (implicit evk: ClassTag[K], evv: ClassTag[V], tracker: ExtKeyTracker) = {
    features.map((kv: (String, Feature)) => {
      val (name, feature) = kv
      val rdd = feature.rdd.asInstanceOf[RDD[(ExtKey[Any], AnyVal)]]
      val steps = tracker.stepsTo(that,rdd).get //throws an exception if rdd is not found
      val grouped = {//if(steps>0) {
          // this rdd-key needs to be dropped backwards to that
        val num = feature.num.asInstanceOf[Numeric[AnyVal]]
        val self = rdd.dropKey(that)
          self.setName("Feature "+name+" on level "+that.name)
          //Todo: use feature-specific instead of general stats parameters
        val stats = feature.stats.getOrElse(selection)
          //get a statsaggregator rdd
        val groupedRdd = statsRdd(self, num).statsPerKey
          groupedRdd.setName("Stats of "+name+" on level "+that.name)
          // get a list of structfields that represent the stats colums
        val struct = selection.map((stat: String) => StructField(name + "_" + stat, statStruct(stat), nullable = true))

        //val dropped = rdd.dropKey(that)
        //toStats(dropped)
          (struct, groupedRdd)
      }// else {
        //  val struct = List(StructField(name, feature.struct, nullable = true))
        //  (struct, rdd)
        //}
      grouped
    })
  }

  def statStruct(stat: String) = {
    stat match {
      case "n" => IntegerType
      case _ => DoubleType
    }
  }

  // case class Feature: rdd, num, name
  // case class Parameter: name, ?

  /**
   * Transforms the rdd to StatsFunctions-rdd. Makes the implicit params explicit,
   * so we can pass a previously captured Numeric-instance.
   * @param self the rdd
   * @param evn Numeric[V] instance
   * @return rdd-to-statsfunctions
   * */
  def statsRdd[K,V <: AnyVal](self: RDD[(ExtKey[K],V)], evn: Numeric[V])
  (implicit evk: ClassTag[K], evv: ClassTag[V]): StatsFunctions[K,V] = {
    rddToStatsFunctions(self)(evk, evv, evn)
  }

  /*def toStats[K](rdd: RDD[(ExtKey[K], AnyVal)])
    (implicit evk: ClassTag[K], tracker: ExtKeyTracker) = {
    new StatsFunctions(rdd).statsPerKey
  } */

  def +=[K,V <: AnyVal](feature: (String, RDD[(ExtKey[K],V)]), stats: Option[List[String]] = Option.empty)
                  (implicit evk: ClassTag[K], evv: ClassTag[V], evn: Numeric[V], tracker: ExtKeyTracker) = {
    val (name, rdd) = feature
    val struct = getType(evv)

    //val persisted = rdd.persist() //This forces a feature to be re-useable without having to recompute.
    // which didn't make any difference in performance ;-)

    //features += name -> rdd
    //numerics += name -> evn
    val res = Feature(rdd, evn, struct, stats)
    features += name -> res
  }

  def getType[V](evv: ClassTag[V]): DataType = {
    val klass = evv.runtimeClass
    // https://spark.apache.org/docs/1.4.0/api/java/org/apache/spark/sql/types/DataType.html
    // ArrayType, BinaryType, BooleanType, DateType, MapType, NullType, NumericType, StringType,
    // StructType, TimestampType, UserDefinedType
    if (classOf[Boolean] isAssignableFrom klass)
      BooleanType
    else if (classOf[java.util.Date] isAssignableFrom klass)
      DateType
    else if (classOf[String] isAssignableFrom klass)
      StringType
    // ByteType, DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType
    else if (classOf[Byte] isAssignableFrom klass)
      ByteType
    //else if (classOf[BigDecimal] isAssignableFrom klass)
    //  DecimalType
    else if (classOf[Double] isAssignableFrom klass)
      DoubleType
    else if (classOf[java.lang.Double] isAssignableFrom klass)
      DoubleType
    else if (classOf[Float] isAssignableFrom klass)
      FloatType
    else if (classOf[Int] isAssignableFrom klass)
      IntegerType
    else if (classOf[java.lang.Integer] isAssignableFrom klass)
      IntegerType
    else if (classOf[Long] isAssignableFrom klass)
      LongType
    else if (classOf[java.lang.Long] isAssignableFrom klass)
      LongType
    else if (classOf[Short] isAssignableFrom klass)
      ShortType
    else
      throw new Exception("Uknown RDD type: "+klass)
  }
}

object Features {
  def getType[V](rdd: RDD[(ExtKey[_],V)])
    (implicit evv: ClassTag[V], tracker: ExtKeyTracker) = {
    val clazz = evv.runtimeClass
    if (classOf[StatisticalSummary] isAssignableFrom clazz)
      "stats"
    else
      "sth"
  }

  def toRow(kv: (ExtKey[_], Array[Iterable[_]]), extra: Int = 0, selection: List[String]) = {
    val (key, columns) = kv
    val v = columns.flatMap((values: Iterable[_]) => {
      if (values.isEmpty)
        //Seq(null)
        Seq.fill(selection.size){null}
      else if (values.size==1) {
        val head = values.head
        head match {
          case stats: ix.extkey.StatsAggregator.C => ix.extkey.Features.statsToSeq(stats, selection)
          //case _ => Seq(head) ++ Seq.fill(selection.size-1){null}
          case _ => Seq(head)
        }
      } else {
        System.err.println("Found more than one value for key: " + key +", values: "+values.mkString("\""))
        throw new SparkException("Found more than one value for key: " + key)
      }
    })
    Row.fromSeq(key.concat().reverse ++ Seq.fill(extra) { null } ++ v)
  }

  def statsToSeq(stats: StatsAggregator.C, selection: Iterable[String] = STATS.keys): Iterable[AnyVal] = {
    for(name <- selection) yield name.toLowerCase match {
      case "mean" => stats.mean
      case "max" => stats.max
      case "min" => stats.min
      case "count" => stats.count
      case "n" => stats.count
      case "std" => stats.sampleStdev
      case "sum" => stats.sum
      case "var" => stats.sampleVariance
      //case unknown => log.error("Unknown stats type requested: "+unknown)
    }
  }

  val STATS = Map(
    "mean" -> DoubleType,
    "max" -> DoubleType,
    "min" -> DoubleType,
    "count" -> IntegerType,
    "n" -> IntegerType,
    "std" -> DoubleType,
    "sum" -> DoubleType,
    "var" -> DoubleType
  )
}