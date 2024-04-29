package org.apache.spark.rdd

import scala._
import org.apache.spark.rdd.ExtKeyFunctions._
import org.scalatest.matchers.ShouldMatchers
import org.scalatest.{FunSpec, GivenWhenThen}
import scala.util.Random
import ix.util.spark.test.SharedSparkContext
import org.apache.spark.SparkContext
import SparkContext._

class ExtKeyTest extends FunSpec
with SharedSparkContext with GivenWhenThen with ShouldMatchers with ExtKeys {

  val n = 10000
  val max = 6

  describe("ExtKeys") {
    it("Should support addition") {
      for (total <- 1 to max) {
        Given("a dataset with " + n + " values")
        val data = createData(n)

        When((total - 1) + " keys are added")
        val rdds = data :: addKey(data, total)

        Then("there should be " + total + " rdds")
        rdds.length should equal(total)

        val keys = factorial(total)
        And("the number of unique keys should be " + keys)
        rdds.last.groupByKey().count should equal(keys)
      }
    }

    it("Should support substraction") {
      for (total <- 1 to max)
        for (dropTo <- 0 until total) {
          Given("a dataset with " + n + " values")
          val data = createData(n)

          When((total - 1) + " keys are added to create a total of " + total + " rdds")
          val rdds = data :: addKey(data, total)
          And("the keys are dropped back till key nr " + (dropTo + 1))
          val dropped = rdds.last.dropKey(rdds(dropTo)) //rdds(x+1) == factorial(x) because the list rdds starts at 0

          val keys = factorial(dropTo + 1)
          Then("the number of unique keys should be " + keys)
          dropped.groupByKey().count should equal(keys)
          And("the number of remaining keys should be " + (dropTo+1))
          val remaining = dropped.map((t: (ExtKey[Int], Int)) => t._1.history.size).stats
          remaining.mean should be (dropTo)
        }
    }

    it("Should support forgetting") {
      for (total <- 1 to max)
        for (take <- 0 until total) {
          Given("a dataset with " + n + " values")
          val data = createData(n)

          When((total - 1) + " keys are added to create a total of " + total + " rdds")
          val rdds = data :: addKey(data, total)
          And("the most recent "+(total-take)+" keys are taken")
          val dropped = rdds.last.takeKey(rdds(take))

          val keys = factorial(total, take+1)
          Then("the number of unique keys should be " + keys)//factorial("+total+","+(take+1)+")=
          dropped.groupByKey().count should equal(keys)
        }
    }
  }

  // Create data
  private def createData(n: Int): RDD[(ExtKey[Int], Int)] = {
    val data = Seq.fill(n)(Random.nextInt(n))
    val start = sc.parallelize(data)
    ext(start.map((value: Int) => Functions.f(value, 1)))
  }

  // Add keys in a recursive way
  private def addKey(prev: RDD[(ExtKey[Int], Int)], max: Int, i: Int = 2): List[RDD[(ExtKey[Int], Int)]] = {
    if (i <= max) {
      val rdd = prev.kvMap((kv: (Int, Int)) => Functions.f(kv._2, i))
      rdd :: addKey(rdd, max, i + 1)
    } else
      List()
  }

  // Compute factorial
  private def factorial(n: Int, i: Int=1): Int =
    if(i>=n) i else i*factorial(n,i+1)
}

// Otherwise Spark raises a non-serializable exception on Scalatest stuff
object Functions {
  def f(value: Int, max: Int): (Int, Int) = (Random.nextInt(max), value)
}