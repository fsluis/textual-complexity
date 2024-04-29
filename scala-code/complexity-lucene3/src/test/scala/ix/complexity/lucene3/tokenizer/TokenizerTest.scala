package ix.complexity.lucene3.tokenizer

import org.scalatest.{Suite, GivenWhenThen, FunSpec}
import org.scalatest.matchers.ShouldMatchers
import de.svenjacobs.loremipsum.LoremIpsum
import org.scalatest.concurrent.Conductors

class TokenizerTest extends FunSpec
with GivenWhenThen with ShouldMatchers with Conductors {
  self: Suite =>
  val data = new LoremIpsum

  describe("Tokenizer") {
    it("Should split text into words") {
      for (n <- List(350, 1000, 10000)) {
        val lorem = data.getWords(n)
        Given("lorem ipsum text of " + n + " words")

        When("the text is tokenized")
        val tokens = Tokenizer.apply(lorem)

        Then("the total amount of words should be " + n)
        tokens.length should be(n)
      }
    }

    it("Should split text into words in multiple concurrent threads") {
      val conductor = new Conductor
      val n = 10000
      val lorem = data.getWords(n)
      for (i <- 1 to 4) {
        conductor.thread("tokenize-" + i) {
          Given("lorem ipsum text of " + n + " words in thread " + Thread.currentThread().getName)

          When("the text is tokenized")
          val tokens = Tokenizer.apply(lorem)

          Then("the total amount of words should be " + n)
          tokens.length should be(n)
        }
      }
      conductor.conduct()
    }

    it("Should remove stopwords") {
      val lorem = data.getWords(350)
      Tokenizer.put("english-stop", Map(
        "language" -> "english",
        "stop" -> true
      ))
      Given("lorem ipsum text of 350 words")

      When("the text is tokenized and stopwords are removed")
      val tokens = Tokenizer.apply(lorem, "english-stop")

      Then("the total amount of words should be less than 350")
      tokens.length should be < 350
    }
  }

  it("Should do stemming") {
    val lorem = data.getWords(350)
    Tokenizer.put("english-stem", Map(
      "language" -> "english",
      "stem" -> true
    ))
    val words = Tokenizer.apply(lorem).map((t: (Int, String)) => t._2.length).toList
    Given("lorem ipsum text of "+words.length+" words containing "+words.sum+" characters")

    When("the text is tokenized and stemmed")
    val stems = Tokenizer.apply(lorem, "english-stem").map((t: (Int, String)) => t._2.length).toList

    Then("the total amount of characters ("+stems.sum+") should be less than "+words.sum)
    stems.sum should be > 0
    stems.sum should be < words.sum
  }
}
