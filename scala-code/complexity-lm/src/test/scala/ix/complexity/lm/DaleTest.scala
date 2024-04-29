package ix.complexity.lm

import org.scalatest.{Suite, GivenWhenThen, FunSpec}
import org.scalatest.matchers.ShouldMatchers
import ix.complexity.lucene3.tokenizer.Tokenizer
import org.scalatest.prop.TableDrivenPropertyChecks

class DaleTest extends FunSpec
with GivenWhenThen with ShouldMatchers with TableDrivenPropertyChecks {
  self: Suite =>
  val sentences = Table(
    // First tuple defines column names
    ("sentence", "dale"),
    // Subsequent tuples define the data
    ("I can almost always tell when movies use fake dinosaurs.", 6),
    ("In 2015 more people have died from taking a #selfie than from shark attacks.", 6)
  )

  describe("Dale") {
    it("Should count Dale words") {
      forAll(sentences) {
        (sentence: String, dale: Int) =>
        val tokens = Tokenizer.apply(sentence).toListMap
        Given("Sentence of " + tokens.size + " tokens")

        When("the dale words are counted")
        val count = Dale2(tokens.values)

        Then("the total number of dale words should be " + dale)
        count should be (dale)
      }
    }
  }
}
