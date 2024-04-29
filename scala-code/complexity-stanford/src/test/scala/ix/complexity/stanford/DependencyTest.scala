package ix.complexity.stanford

import org.scalatest.{Suite, GivenWhenThen, FunSpec}
import org.scalatest.matchers.ShouldMatchers
import ix.complexity.lucene3.tokenizer.Tokenizer
import org.scalatest.prop.TableDrivenPropertyChecks
import edu.stanford.nlp.trees.TypedDependency
import scala.collection.JavaConversions._
import edu.stanford.nlp.ling.TaggedWord
import ix.complexity.features.sentence.{StanfordDLT}

class DependencyTest extends FunSpec
with GivenWhenThen with ShouldMatchers with TableDrivenPropertyChecks {
  self: Suite =>
  val sentences = Table(
    // First tuple defines column names
    ("sentence", "dlt"),
    // Subsequent tuples define the data
    ("I can almost always tell when movies use fake dinosaurs.", 2),
    ("In 2015 more people have died from taking a #selfie than from shark attacks.", 0),
    ("The horse raced past the barn fell.", 1)
  )

  describe("Dependency") {
    it("Should find dependencies and calculate DLT") {
      forAll(sentences) {
        (sentence: String, target: Int) =>
        val tokens = Tokenizer.apply(sentence).toListMap

        Given("Sentence of " + tokens.size + " tokens")

        When("the tokens are tagged")
        val tagged = Tagger(tokens)

        And("the grammar is parsed")
        val grammar = Dependencies(tagged.values.toSeq)

        Then("there should be dependencies")
        val dependencies = grammar.typedDependenciesCollapsed()
        dependencies.size should be > 0

        And("the total DLT should be " + target)
        StanfordDLT.computeDLT(tagged, grammar).sum should be (target)
      }
    }
  }

}
