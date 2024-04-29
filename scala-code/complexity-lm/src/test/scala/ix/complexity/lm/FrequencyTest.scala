package ix.complexity.lm

import org.scalatest.{Suite, GivenWhenThen, FunSpec}
import org.scalatest.matchers.ShouldMatchers
import ix.complexity.lucene3.tokenizer.Tokenizer
import org.scalatest.prop.TableDrivenPropertyChecks

class FrequencyTest extends FunSpec
with GivenWhenThen with ShouldMatchers with TableDrivenPropertyChecks {
  self: Suite =>
  val sentences = Table(
    // First tuple defines column names
    ("sentence", "frequency"),
    // Subsequent tuples define the data
    ("I can almost always tell when movies use fake dinosaurs.", 7.05d),
    ("In 2015 more people have died from taking a #selfie than from shark attacks.", 7.49d)
  )

  describe("Dale") {
    it("Should retrieve frequencies") {
      forAll(sentences) {
        (sentence: String, frequency: Double) =>
        val tokens = Tokenizer.apply(sentence).map(_._2).toList
        Given("Sentence of " + tokens.size + " tokens")

        When("the frequencies are retrieved")
        val frequencies = WordFrequencies(tokens)
        println(frequencies)

        Then("the total number of frequencies should be " + tokens.size)
        frequencies.size should be (tokens.size)

        And("the average frequency should be "+frequency)
        val filtered = frequencies.filter(!_.isNaN)
        filtered.sum / filtered.size should be (frequency +- 0.1)
      }
    }
  }
}
