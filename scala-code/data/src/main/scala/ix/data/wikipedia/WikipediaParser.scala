package ix.data.wikipedia

import java.util
import java.util.{ArrayList, List}
import java.util.regex.Pattern

import com.google.common.base.Splitter
import com.google.common.collect.Iterables
import de.tudarmstadt.ukp.wikipedia.parser.mediawiki.{FlushTemplates, MediaWikiParser, MediaWikiParserFactory}
import de.tudarmstadt.ukp.wikipedia.parser.{Content, ContentElement, LinksInterface, Paragraph, ParsedPage, Section, Span}
import ix.data.structure.{Text, ContentList}
import ix.util.services2.{ModelService, Models}
import org.apache.commons.configuration.Configuration
import org.apache.commons.lang3.{StringEscapeUtils, StringUtils}
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.util.matching.Regex

object WikipediaParser extends ModelService[WikipediaParser] {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  if (!configurations.contains(Models.DEFAULT))
    put(Models.DEFAULT, Map[String,Nothing]())
  put("flush", Map[String,Nothing]())

  def load(name: String, confOpt: Option[Configuration]): WikipediaParser = {
    val clean = confOpt.forall(_.getBoolean("clean", true))
    val includeMath = confOpt.forall(_.getBoolean("includeMath", false))

    val pf = new MediaWikiParserFactory
    name match {
      case "flush" => pf.setTemplateParserClass( classOf[FlushTemplates] )
      case _ =>
    }
    pf.setShowMathTagContent(includeMath)
    new WikipediaParser(pf.createParser, clean)
  }

  def parse(wikiText: String, name: String = Models.DEFAULT): ParsedPage =
    get(name).cleanAndParseWiki(wikiText)

  @deprecated("Use apply() instead")
  def sections(page: ParsedPage): mutable.Buffer[(Int,Section)] =
    page.getSections.zipWithIndex.map(_.swap)

  @deprecated("Use apply() instead")
  def paragraphs(section: Section): mutable.Buffer[(Int,String)] = {
    val options = for(par <- section.getParagraphs) yield Option(TextParser.parse(par))
    // flatten removes None's / nulls, filter removes empty strings
    options.flatten.filter(_.trim.nonEmpty).zipWithIndex.map(_.swap)
  }

  def apply(wikiText: String, modelName: String = Models.DEFAULT) = {
    val m = get(modelName)
    val page = m.cleanAndParseWiki(wikiText)
    var charIndex = 0
    val ss = for(s <- page.getSections) yield {
      val pp = for(p <- s.getParagraphs) yield {
        val content = Text(m.extractText(p).trim, charIndex)
        charIndex += content.plainText.length
        content
      }
      //val pp = section.getParagraphs.map(p => Text(m.extractText(p), p.getSrcSpan.getStart) )
      // getSrcSpan is always null...
      new ContentList( pp.filter(_.plainText.nonEmpty) )
    }
    new ContentList( ss.filter(_.sorted.nonEmpty) )
  }
}

class WikipediaParser(val wikiParser: MediaWikiParser, val clean: Boolean) {
  val math: Regex = """(?s)<math>(.*?)<\/math>""".r
  val CATEGORY = """\[\[Category:.+\]\]""".r
  val LANG = """\[\[[a-z\-]{2,10}:.+\]\]""".r
  val GOT = """\[\[got:$""".r
  val tags = Seq(CATEGORY, LANG, GOT)
  val SPLITTER = Splitter.on("|")

  /*
  [[Category: ...]] should be removed beforehand, parser doesn't deal with it fully
  Same for LANG, sometimes parser leaves weird some weird languages
  fx lang is [[zh-min-nan: ...]]
  GOT? I don't know...
  Math also needs pre removal

  original Image/File ones get handled as links here
   */

  import WikipediaParser._

  def parse(wikiText: String): ParsedPage = cleanAndParseWiki(wikiText)

  def cleanWiki(wikiText: String) = {
    var cleaned = wikiText
    if (clean) {
      for(tag <- tags)
        cleaned = tag.replaceAllIn(cleaned, "")
      //cleaned = removeMath(cleaned)
      // there's a toggle on the MediaWikiParserFactory to exclude <math> content
    }
    cleaned
  }

  def cleanAndParseWiki(wikiText: String): ParsedPage = {
    wikiParser.parse(cleanWiki(wikiText))
  }

  def removeMath(wikiText: String): String = {
    if (StringUtils.countMatches(wikiText, "<math>")==StringUtils.countMatches(wikiText, "</math>"))
      return math.replaceAllIn(wikiText, "")
    else
      log.warn("Skipping math removal for page, unequal counts for opening and closing tags.")
    wikiText
  }

  /**
   * String escaping and link parsing for Section and Paragraph
   * Copied from old Java code
   * @param p Either Paragraph, Section, or ParsedPage
   * @return plain text
   */
  def extractText(p: LinksInterface): String = {
    var text: String = p.getText
    if(clean) {
      text = StringEscapeUtils.unescapeHtml4(text)

      val links = new util.ArrayList[Span]
      import scala.collection.JavaConversions._
      for (link <- p.getLinks) {
        links.add(link.getPos)
      }
      val linkText = p.getText(links)
      //System.out.println(linkText.length() + "/" + text.length());
      if (linkText.length >= text.length) return ""

      // added 19-10-2019
      import scala.collection.JavaConversions._
      for (link <- p.getLinks) {
        val link_text = link.getText
        val anchor_text = Iterables.getLast(SPLITTER.split(link_text))
        text = text.replace(link_text, anchor_text)
      }
    }
    text
  }

}