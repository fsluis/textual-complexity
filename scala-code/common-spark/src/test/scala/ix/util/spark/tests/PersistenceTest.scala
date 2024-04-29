package ix.util.spark.tests

import ix.util.spark.test.{SharedSparkContext, DummyService}
import scala.Predef._
import org.apache.spark.SparkContext._
import scala._

//import org.apache.spark.extkey.Preamble._
//import scala.Predef.String
import scala.util.Random
import org.scalatest.{run, GivenWhenThen, FunSpec}
import org.scalatest.matchers.ShouldMatchers

/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 12/1/13
 * Time: 9:48 PM
 * To change this template use File | Settings | File Templates.
 */
class PersistenceTest extends FunSpec
with SharedSparkContext with GivenWhenThen with ShouldMatchers {
  val n = 1000000
  val partitions = 4 //per executor
  val data = Seq.fill(n)(Random.nextInt(n))

  describe("Singleton services") {
    it("Should be warmed up") {
      Given("A dataset with "+n+" values")

      When("the data is distributed over "+partitions+" partitions")
      val start = sc.parallelize(data, partitions)

      Then("the distributed data should have "+n+" values")
      start.count should be (n)

      And("there should be at least one executor")
      getExecInfo().length should be > (0)

      println(getExecInfo().mkString("\n"))
    }

    it("Should have distinct hashcodes for each executor") {
      val executors = 1.max(getExecInfo().length)
      Given(executors+" executors and a dataset with "+n+" values")

      When("the data is processed in "+executors*partitions+" partitions")
      val start = sc.parallelize(data, executors*partitions)

      And("the hashcodes of the DummyService are collected")
      val master = DummyService.hashCode()
      val codes = start.map((i: Int) => PersistenceTest.f(i))

      Then("there should be "+executors+" distinct hashcodes")
      val distinct = codes.distinct().collect().toList //.distinct
      distinct.length should be (executors)
      println("distinct 1: "+distinct)

      if("local"!=uri) {
        And("the master's hashcode should be different from the workers' codes")
        distinct should not contain (master)
      }
    }

    it("Should be created by different threads") {
      val executors = 1.max(getExecInfo().length)
      Given(executors+" executors and a dataset with "+n+" values")

      When("the data is processed in "+executors*partitions+" partitions")
      val start = sc.parallelize(data, executors*partitions)

      And("the thread names of the DummyService are collected")
      val master = DummyService.startedByThread
      val threads = start.map((i: Int) => PersistenceTest.g(i))

      Then("there should be "+executors+" distinct thread names")
      val distinct = threads.distinct().collect().toList //.distinct
      println("distinct 2: "+distinct)
      distinct.length should be (executors)

      if("local"!=uri) {
        And("the master's thread name should be different from the workers' thread names")
        distinct should not contain (master)
      }
    }

    it("Should not share a jvm") {
      val executors = 1.max(getExecInfo().length)
      Given(executors+" executors and a dataset with "+n+" values")

      When("the data is processed in "+executors*partitions+" partitions")
      val start = sc.parallelize(data,executors*partitions)

      And("the DummyService jvm process ids and hashcodes are collected")
      val master = DummyService.jvm -> List(DummyService.hashCode)
      val ids = start.map((i: Int) => PersistenceTest.k(i))
      val codes = ids.map((s: String) => PersistenceTest.j(s))

      Then("there should be 1 hashcode per jvm")
      // Distinct thread names per hashcode
      val distinct = codes.groupByKey().map((kv: (String, Iterable[Int])) => PersistenceTest.l(kv._1,kv._2))
      val mapped = distinct.collect().toMap + master
      val avg = mapped.map((kv: (String, List[Int])) => kv._2.length).sum / mapped.size
      //val distinct = (master :: threads.distinct().collect().toList).distinct
      println("mapped 3: "+mapped)
      //println(avg)
      avg should be (1)
    }
  }
}

object PersistenceTest extends App {
  override def main(args: Array[String]): Unit = {
    //(new PersistenceTest).execute()
    run(new PersistenceTest)
  }

  def f(i: Int): Int =
    DummyService.hashCode()

  def g(i: Int): String =
    DummyService.startedByThread

  def h(i: Int): (Int, String) =
    (i, DummyService.startedByThread)

  def i(a: Int, b: Seq[String]): (Int, List[String]) =
    (a, b.toList.distinct)

  def j(a: String): (String, Int) =
    (a, DummyService.hashCode)

  def k(i: Int): String =
    DummyService.jvm

  def l(a: String, b: Iterable[Int]): (String, List[Int]) =
    (a, b.toList.distinct)
}

