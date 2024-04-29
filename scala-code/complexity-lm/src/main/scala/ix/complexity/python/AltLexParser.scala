package ix.complexity.python

import java.io.IOException

import net.liftweb.json.{DefaultFormats, JsonParser}
import net.liftweb.json.JsonParser.ParseException

/**
 * Created by f on 04/02/2020.
 */
object AltLexParser extends ProcessService {
  implicit val formats = new DefaultFormats {}

  def parse(name: String, text: String): Option[AltLexResult] = {
    val option = analyze(name, text)
    try {
      option.map(JsonParser.parse(_).extract[AltLexResult])
    } catch {
      case pe: ParseException => { ProcessService.log.error("Couldn't parse: "+option, pe); None }
    }
  }
}

case class AltLexResult(nrOfWords: Int, nrOfCausalAltlexes: Int, nrOfSentences: Int, nrOfTokens: Int, causalWordSpan: Int, nrOfNonCausalAltlexes: Int, nonCausalWordSpan: Int) {
  def +(that: AltLexResult) = AltLexResult(this.nrOfWords + that.nrOfWords, this.nrOfCausalAltlexes + that.nrOfCausalAltlexes, this.nrOfSentences + that.nrOfSentences, this.nrOfTokens + that.nrOfTokens, this.causalWordSpan + that.causalWordSpan, this.nrOfNonCausalAltlexes+ that.nrOfNonCausalAltlexes, this.nonCausalWordSpan + that.nonCausalWordSpan)
}
