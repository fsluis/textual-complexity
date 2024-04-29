package ix.complexity.features.common

import org.scalatest.matchers.ShouldMatchers
import org.scalatest.prop.TableDrivenPropertyChecks
import org.scalatest.{Suite, GivenWhenThen, FunSpec}
import com.google.common.collect.{ImmutableList, HashMultiset, Multiset}
import scala.util.Random
import scala.collection.JavaConversions._

// compare jaguar with jaguar, jaguar with bmw, and jaguar with bike, and jaguar with beer?

class EntropyTest extends FunSpec
with GivenWhenThen with ShouldMatchers with TableDrivenPropertyChecks {
  self: Suite =>

  val symbols = Array("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")

  val bags = Table(
    ("distribution", "entropy"), // First tuple defines column names
    (Array(1,1), 1d), // Subsequent tuples define the data
    (Array(1,1,1), 1.58d),
    (Array(1,2), .92d)
  )
  val lists = Table(
    ("repetitions", "window"), // First tuple defines column names
    //(25, 25),
    (100, 25),
    (1000, 25),
    (10000, 25),
    (100000, 25)
  )

  describe("Entropy") {
    it("Should correctly calculate entropy of a bag of character symbols") {
      forAll(bags) {
        (distribution: Array[Int], target: Double) =>
          Given("a distribution of [" + distribution.mkString(",") + "]")

          When("entropy is calculated over a bag of symbols")
          val bag = createBag(distribution)
          val entropy = JavaEntropy.calculateEntropy(bag).entropy

          Then("entropy should be %2.2f".format(target))
          entropy should be(target plusOrMinus .01)
      }
    }
  }

  describe("Sliding Window Entropy (SWE)") {
    it("Should correctly calculate entropy of a bag of character symbols") {
      forAll(bags) { (distribution: Array[Int], target: Double) =>
        forAll(lists) { (repetitions: Int, window: Int) =>
          Given("a distribution of [" + distribution.mkString(",") + "] that is repeated " + repetitions+ " time")

          When("swe is calculated over a window of "+window)
          val bag = createBag(distribution, repetitions)
          val sequence = createSequence(bag)
          val entropies = SlidingWindowEntropy.calculateEntropy(sequence, 1, window)

          Then("swe should be %2.2f +/- %2.2f".format(target, entropies(0).getStandardDeviation))
          entropies(0).getMean should be(target plusOrMinus entropies(0).getStandardDeviation)

          //println("entropy: "+entropies(0).getMean+"; target: "+target+"; values: "+entropies(0).getN+"; variance: "+entropies(0).getVariance+"; sd: "+entropies(0).getStandardDeviation)
        }
      }
    }
  }

  def createBag(distribution: Array[Int], repetitions:Int = 1): Multiset[String] = {
    val bag: Multiset[String] = HashMultiset.create()
    for (i <- 0 until distribution.size)
      bag.add(symbols(i), distribution(i) * repetitions)
    bag
  }

  def createSequence(bag: Multiset[String]): Seq[String] = {
    val list: java.util.List[String] = ImmutableList.copyOf(bag.iterator())
    Random.shuffle(list)
  }
}