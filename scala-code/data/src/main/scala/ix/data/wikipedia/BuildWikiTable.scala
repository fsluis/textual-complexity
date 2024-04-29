package ix.data.wikipedia

import de.tudarmstadt.ukp.wikipedia.parser.ParsedPage
import de.tudarmstadt.ukp.wikipedia.parser.mediawiki.MediaWikiParserFactory
import ix.common.data.wikipedia.WikiTables
import ix.util.net2.jdbc.MySQLDatabase
import org.apache.commons.logging.{LogFactory, Log}
import scala.collection.JavaConversions._
import org.apache.hadoop.io.LongWritable

class BuildWikiTable {
  //Sink database
  val database = MySQLDatabase("wiki_pages", "supermarkt").get
  //val pages = new WikiTables.PagesTable(database, "nl2")
  val pages = new WikiTables.PagesTable(database, "nl_20160305")
  pages.create

  //Dutch wiki
  //val wiki = WikipediaService.get("nl2")
  val wiki = WikipediaService.get("dutch-20160305")

  //Parser
  val parser = WikipediaParser.get("flush")

  def isStub(pp: ParsedPage): String = {
    for (template <- pp.getTemplates)
      if (template.getName.contains("stub")) return template.getName
    null
  }

  def run() = {
    val articles = wiki.getArticles.iterator()
    var counter = 0
    while(articles.hasNext) {
      val article = articles.next()
      counter += 1
      try {
        val page = parser.parse(article.getText)

        if (page == null)
          BuildWikiTable.log.info("Empty page nr. "+counter+", text: " + article.getText)
        else {
          if (counter % 10000 == 0) BuildWikiTable.log.info("Getting wiki pages nr. " + counter)

          pages.addPage(
            "dutch",
            article.getPageId,
            article.getTitle.getPlainTitle,
            isStub(page),
            article.isDisambiguation,
            article.isRedirect,
            page.length(),
            null,
            null,
            0,
            article.getText
          )
        }
      } catch {
        case e: Exception => BuildWikiTable.log.error("Error while parsing page nr. "+counter, e)
      }
    }
  }

  def close() = {}
}



object BuildWikiTable extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  override def main(args: Array[String]): Unit = {
    val builder = new BuildWikiTable
    try {
      builder.run()
    }
    finally {
      builder.close()
    }
  }
}