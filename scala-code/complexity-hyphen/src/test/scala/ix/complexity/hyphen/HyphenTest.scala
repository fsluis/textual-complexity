package ix.complexity.hyphen

import _root_.de.svenjacobs.loremipsum.LoremIpsum
import ix.complexity.lucene3.tokenizer.Tokenizer
import org.scalatest.matchers.ShouldMatchers
import org.scalatest.prop.TableDrivenPropertyChecks
import org.scalatest.{FunSpec, GivenWhenThen, Suite}

// compare jaguar with jaguar, jaguar with bmw, and jaguar with bike, and jaguar with beer?

class HyphenTest extends FunSpec
with GivenWhenThen with ShouldMatchers with TableDrivenPropertyChecks {
  self: Suite =>

  val words = Table(
    ("dict", "word", "syllables"), // First tuple defines column names
    ("english", "syllables", 2), // Subsequent tuples define the data
    ("english", "model", 2),
    ("english", "word", 1),
    ("english", "notwithstanding", 4),
    ("english", "over", 1),
    ("dutch", "desalniettemin", 5),
    ("dutch", "daarover", 2),
    ("dutch", "over", 1),
    ("dutch", "woord", 1),
    ("dutch", "model", 2)
  )

  val data = new LoremIpsum
  import Hyphenator.SEP

  describe("Hyphenation") {
    it("Should detect correct number of syllables") {
      forAll(words) {
        (model: String, word: String, target: Int) =>
          Given("hyphenator model " + model + " and word " + word)
          val dict = Hyphenator.get(model)

          When("the word is hyphenated and syllables counted")
          //val syllables = dict.syllables(word)
          //val hyph = dict.hyphenate(word)
          val hyph = dict.hyphenate(word, SEP, SEP)
          val syllables = hyph.split(SEP)
          println("word: "+word+"; hyph: "+hyph+"; syl: "+syllables)

          Then("the number of syllables should be " + target)
          syllables.size should be(target)
      }
    }

    it("Should detect syllables in a bulk of words") {
      for (n <- List(350, 1000, 10000, 100000, 1000000)) {
        val model = "english"
        val lorem = data.getWords(n)
        val dict = Hyphenator.get(model)
        Given("hyphenator model " + model + " lorem ipsum text of " + n + " words")

        When("the text is tokenized")
        val tokens = Tokenizer.apply(lorem).toList
        And("the syllables are extracted")
        val hyphs = tokens.map((t: (Int, String)) => dict.hyphenate(t._2, SEP, SEP).split(SEP).length)

        Then("the total amount of hyphs ("+hyphs.sum+") should be more than " + tokens.length)
        hyphs.sum should be > (tokens.length)
      }
    }

  }
}