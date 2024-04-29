package ix.complexity.stanford

import org.scalatest.{Suite, GivenWhenThen, FunSpec}
import org.scalatest.matchers.ShouldMatchers
import ix.complexity.lucene3.tokenizer.Tokenizer
import org.scalatest.prop.TableDrivenPropertyChecks
import scala.collection.immutable.ListMap

class TaggerTest extends FunSpec
with GivenWhenThen with ShouldMatchers with TableDrivenPropertyChecks {
  self: Suite =>
  val sentences = Table(
    // First tuple defines column names
    ("sentence", "nouns", "verbs"),
    // Subsequent tuples define the data
    ("I can almost always tell when movies use fake dinosaurs.", 2, 2),
    ("In 2015 more people have died from taking a #selfie than from shark attacks.", 4, 3)
  )

  describe("Tagger") {
    it("Should tag words") {
      forAll(sentences) {
        (sentence: String, nouns: Int, verbs: Int) =>
        val tokens = Tokenizer.apply(sentence).toListMap
        Given("Sentence of " + tokens.size + " tokens")

        When("the tokens are tagged")
        val tagged = Tagger(tokens)
        val tags = tagged.values.map(_.tag)

        //System.out.println("Tokens: "+tokens.values+" ("+tokens.size+")")
        //System.out.println("Tagged: "+tagged.values+" ("+tagged.size+")")

        Then("the total amount of tags should be " + tokens.size)
        tagged.values.size should be (tokens.size)

        And("there should be " + nouns + " nouns")
          tags.count(_.startsWith("NN")) should be (nouns)

        And("there should be " + verbs + " verbs")
          tags.count(_.startsWith("VB")) should be (verbs)

      }
    }
  }
}
