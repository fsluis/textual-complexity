package ix.complexity.lucene3.esa.test

import ix.complexity.lucene3.esa.{CosineKernel, EsaGeneralityService, EsaService}
import ix.complexity.lucene3.tokenizer.Tokenizer
import org.scalatest.matchers.ShouldMatchers
import org.scalatest.prop.TableDrivenPropertyChecks
import org.scalatest.{Suite, GivenWhenThen, FunSpec}
import ix.complexity.features.word.EsaFeatures
import gnu.trove.map.hash.TIntDoubleHashMap
import scala.collection.JavaConversions._

class EsaTest extends FunSpec
with GivenWhenThen with ShouldMatchers with TableDrivenPropertyChecks {
  self: Suite =>

  val comparisons = Table(
      ("a", "b", "target"), // First tuple defines column names
      ("jaguar", "jaguar", 1d), // Subsequent tuples define the data
      ("jaguar", "bmw", -1d),
      ("jaguar", "bike", -1d),
      ("jaguar", "beer", 0d)
    )
  val words = List(
      "science",
      "psychology",
      "cognitive psychology",
      "artificial intelligence"
    )
  val sentences = List(
    "A collection of implicit conversions supporting interoperability between Scala and Java collections",
    "Audi is a German automobile manufacturer that designs, engineers, manufactures and distributes automobiles",
    "The Netherlands is a constituent country of the Kingdom of the Netherlands"
  )
  val cwords = List("car", "bike", "engine", "beer")
  val esaModel = "default"
  val esaGeneralityModel = "default"
  val tokenizer = "english-all"

  describe("ESA") {
    it("Should find and compare concepts of decreasing similarity") {
      var last = 0d
      forAll(comparisons) {
        (a: String, b: String, target: Double) =>
          Given("esa model " + esaModel + " and words " + a + " and " + b)
          val esa = EsaService.get()

          When("the words are tokenized and concepts retrieved and compared")
          val tokens = Tokenizer.apply(a + " " + b, tokenizer).map((t: (Int, String)) => t._2).toList
          val sim = esa.getRelatedness(tokens(0), tokens(1))
          //val concepts = tokens.map((t: (Int, String)) => esa.getConceptMap(t._2))

          if (target < 0) {
            Then("the cosine similarity should be less than " + last)
            sim should be < (last)
            And("the cosine similarity should be at least 0")
            sim should be >= (0d)
          } else {
            Then("the cosine similarity should be " + target)
            sim should be(target plusOrMinus .01)
            if (last > 0) {
              And("the cosine similarity should be less than " + last)
              sim should be < (last)
            }
          }
          last = sim
        //println("words: "+a+" and "+b+"; sim: "+sim)
      }
    }
  }

  describe("ESA generality") {
    it("Should find and compare concepts of decreasing generality") {
      var last = 0d
      for (word <- words) {
        Given("esa model " + esaModel + ", generality model " + esaGeneralityModel + " and word " + word)

        When("the words are tokenized and concepts retrieved and compared")
        val token = Tokenizer.apply(word, tokenizer).map((t: (Int, String)) => t._2).next
        val topic = EsaService.topic(token, esaModel)
        val generality = EsaGeneralityService.generality(topic, esaGeneralityModel)

        Then("the generality should be more than 0")
        generality should be > (0d)
        if (last > 0) {
          Then("the generality should be less than " + last)
          generality should be < (last)
        }
        last = generality
        //println("word: "+word+"; generality: "+generality+"; concept size: "+topic.size()+"; concept sum: "+topic.values.sum)
      }
    }
  }

  describe("ESA sliding window centroid") {
    it("Should compute centroid over a sliding window of concepts") {
      for (window <- 1 to words.size) {
        val tokens = Tokenizer.apply(words.mkString(" "), tokenizer).map((t: (Int, String)) => t._2).toList
        Given("esa model " + esaModel + ", window size " + window + " and " + tokens.size + " tokens")

        When("the concepts are retrieved and a sliding window is applied")
        val topics = EsaFeatures.topics(tokens.iterator, esaModel)
        val centroids = EsaFeatures.centroids(topics, window)

        Then("the number of centroids should be " + (tokens.size - window + 1))
        centroids.size should be(tokens.size - window + 1)
      }
    }
  }

  describe("ESA centroid entropy") {
    it("Should find and compare concepts of decreasing entropy") {
      var last = 0d
      for(sentence <- sentences) {
        Given("esa model "+esaModel+" and sentence: "+sentence.take(15)+"...")

        When("the words are tokenized and concepts retrieved and compared")
        val tokens = Tokenizer.apply(sentence, tokenizer).map((t: (Int, String)) => t._2)
        val topics = EsaFeatures.topics(tokens, esaModel)
        val centroid = EsaFeatures.centroid(topics)
        val entropy = EsaFeatures.entropy(centroid)

        Then("the entropy (%2.2f) should be more than 0".format(entropy))
        entropy should be > (0d)
        if(last>0) {
          And("the entropy should be less than "+last)
          entropy should be < (last)
        }
        last = entropy
        //println("sentence: "+sentence+"; entropy: "+entropy)
      }
    }
  }

  describe("ESA cohesion") {
    it("Should find and compare concepts of decreasing cohesion") {
        Given("esa model "+esaModel+" and words: "+cwords.mkString(", ").take(15)+"...")

        When("the words are tokenized and concepts retrieved")
        val tokens = Tokenizer.apply(cwords.mkString(" "), tokenizer).map((t: (Int, String)) => t._2)
        val topics = EsaFeatures.topics(tokens, esaModel).toIterable
        val cohesion = EsaFeatures.cohesion(topics, weigh=false)
      //ToDo: make this referencefeatures thing insightful!

      /*val kernel = new CosineKernel
      val matrix = kernel.matrix(asJavaIterable(topics))
      for(i <- 0 until matrix.numRows())
        for(j <- 0 until matrix.numColumns())
          println("["+i+","+j+"]="+matrix.get(i,j))

      println("words: "+cwords.mkString(",")+"; cohesion: "+cohesion.mkString(","))*/

      for(i <- 1 until cwords.size) {
        Then("cohesion_"+i+" (%5.5f) should be more than 0".format(cohesion(i)))
        cohesion(i) should be > (0d)
        And("cohesion_"+i+" should be more than cohesion_"+(i+1)+" (%5.5f)".format(cohesion(i+1)))
        cohesion(i) should be > (cohesion(i+1))
      }
    }
  }
}