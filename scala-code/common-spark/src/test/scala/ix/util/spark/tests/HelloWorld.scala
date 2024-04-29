package ix.util.spark.tests

/*** SimpleApp.scala ***/

import ix.util.spark.test.SharedSparkContext
import org.apache.spark.SparkContext._
import org.scalatest.{Suite, run, GivenWhenThen, FunSpec}
import org.scalatest.matchers.ShouldMatchers
import scala.collection.JavaConversions._
import de.svenjacobs.loremipsum.LoremIpsum

class HelloWorld extends FunSpec
with SharedSparkContext with GivenWhenThen with ShouldMatchers {
  self: Suite =>
  val data = new LoremIpsum

  describe("Hello (Spark) world") {
    it("should count words") {
      //println(data.getWords(350))
      val lorem = data.getWords(350).split("""\. """)
      Given("lorem ipsum of 350 words splitted into "+lorem.size+" sentences")

      When("the words are counted")
      val sentences = sc.parallelize(lorem)
      val words = sentences.flatMap((line: String) => line.split(' '))
      val ones = words.map((word: String) => (word, 1))
      val counts = ones.reduceByKey(_ + _)

      Then("the total amount of words should be 350")
      val total = counts.values.sum
      total should be (350)
      And("the most popular word (\"et\") should have 28 observations")
      val map = Map(counts.collect():_*)
      map.get("et") should be (Some(28))
      //counts.collect().foreach(p => println(p._1+": "+p._2))
    }
  }
}

object HelloWorld extends App {
  override def main(args: Array[String]): Unit = {
    //(new HelloWorld).execute()
    run(new HelloWorld)
  }
}
