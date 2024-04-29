package ix.data.com.datasets

import de.tudarmstadt.ukp.wikipedia.api.{DatabaseConfiguration, Page, WikiConstants, Wikipedia}
import de.tudarmstadt.ukp.wikipedia.parser.mediawiki.MediaWikiParserFactory
import ix.common.util.TimeLogger
import ix.data.wikipedia.WikipediaParser
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import ix.util.net2.jdbc.Flushable

import scala.collection.mutable
import scala.util.Try
import ix.complexity.lucene3.tokenizer.{Tokenizer => LuceneTokenizer}
import ix.complexity.stanford.{Annotator => StanfordAnnotator}

// java -cp "$HOME/farm/lib/*:$HOME/farm/bin/spark-2.4.3-bin-hadoop2.7/jars/*" ix.data.com.datasets.WikiPairsTable
object WikiPairsTable extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)
  lazy val tlog = new TimeLogger(log, "per pair", 10000)
  lazy val enwiki = newEnWiki()
  lazy val siwiki = newSiWiki()
  lazy val sidb = ix.util.net2.jdbc.MySQLDatabase("kommlabir01fl", "wiki_simple_20200401").get
  lazy val ensidb = ix.util.net2.jdbc.MySQLDatabase("kommlabir01fl", "ensiwiki_2020").get

  //Parser
  lazy val pf = new MediaWikiParserFactory
  lazy val parser = pf.createParser

  override def main(args: Array[String]): Unit = {
    run()
  }

  def run(): Unit = {

    // Prepare langlinks
    val query = sidb.connection.prepareStatement("SELECT ll_from AS simple_id, ll_title AS english_title FROM langlinks WHERE ll_lang='en'")
    val res = query.executeQuery()
    val langlinks = scala.collection.mutable.Map[Int, String]()
    while(res.next()) {
      langlinks(res.getInt("simple_id")) = res.getString("english_title")
    }

    // Keep track of english ids, as we will find multiple pages pointing to the same English page
    // and want to add only once to the info table
    val enIds = mutable.Set[Int]()

    // Prepare pairs table
    ensidb.connection.prepareStatement("DROP TABLE IF EXISTS pairs").execute()
    ensidb.connection.prepareStatement("CREATE TABLE pairs ( id BIGINT NOT NULL AUTO_INCREMENT, simple_id BIGINT NOT NULL, english_id BIGINT NOT NULL, PRIMARY KEY (id), UNIQUE INDEX ids (simple_id,english_id))").execute()
    val pairsSink = new Flushable(ensidb.connection, "INSERT INTO pairs (simple_id, english_id) VALUES", 2, 1000, true)

    // Prepare pages_info table
    ensidb.connection.prepareStatement("DROP TABLE IF EXISTS pages_info").execute()
    ensidb.connection.prepareStatement("CREATE TABLE pages_info ( id BIGINT NOT NULL AUTO_INCREMENT, page_id BIGINT NOT NULL, lang VARCHAR(10) NOT NULL, title TEXT COLLATE utf8mb4_general_ci, is_disambiguation TINYINT, is_discussion TINYINT, redirect_count INT, redirects MEDIUMTEXT COLLATE utf8mb4_general_ci, stub_count INT, stubs TEXT COLLATE utf8mb4_general_ci, lucene_word_count INT, stanford_sentence_count INT, PRIMARY KEY (id), UNIQUE INDEX wiki_id (page_id, lang))").execute()
    val infoSink = new Flushable(ensidb.connection, "INSERT INTO pages_info (lang, page_id, title, is_disambiguation, is_discussion, redirect_count, redirects, stub_count, stubs, lucene_word_count, stanford_sentence_count) VALUES", 11, 1000, true)

    // Simple page iterator
    val siPages = siwiki.getArticles.iterator()
    var (tried, found) = (0,0)
    while(siPages.hasNext ) {
      val t = tlog.start()
      val siPage = siPages.next()

      try {
        // Get simple info
        val siInfo = parsePageInfo(siPage)
        infoSink.addDeliveries("simple" +: siInfo.productIterator.toSeq)

        // Find english page
        val englishTitle = langlinks.get(siPage.getPageId)
        val enPageOption = englishTitle.flatMap( title => Try(enwiki.getPage(title)).toOption )
        val isNewId = enPageOption.exists( page => enIds.add(page.getPageId) ) //does a page exists that was added to the set of ids?

        // Some rolling statistics
        tried+=1
        if(enPageOption.isDefined) found+=1
        if(tried%10000==0)
        log.info("Tried to search "+tried+" pairs, found "+found+" pairs")

        // Parse and sink english info
        if(isNewId) enPageOption.map(parsePageInfo).foreach(enInfo => infoSink.addDeliveries( "english" +: enInfo.productIterator.toSeq ) )

        // Sink pairs
        enPageOption.map(_.getPageId).foreach( enPageId => pairsSink.addDeliveries( Seq(siPage.getPageId, enPageId) ) )
      } catch {
        case e: Exception => log.warn("Error while processing page " + siPage.getPageId)
      }
      tlog.stop(t)
    }

    log.info("Finishing with "+tried+" searches and "+found+" pairs found")
    pairsSink.finish()
    infoSink.finish()
    log.info("Finished")
  }

  def parsePageInfo(page: Page): PageInfo = {
    val pp = parser.parse(page.getText)
    val stubs = pp.getTemplates.map(_.getName).filter(_.contains("stub"))
    val counts = parsePageCounts(page)
    //val redirectIds = page.getRedirects.flatMap(siwiki.getPageIds(_).toTraversable)
    PageInfo(page.getPageId, page.getTitle.getPlainTitle, page.isDisambiguation, page.isDiscussion, page.getRedirects.size, page.getRedirects.mkString(", "), stubs.size, stubs.mkString(", "), counts._1, counts._2)
  }

  def parsePageCounts(page: Page) = {
    val content = WikipediaParser(page.getText, "flush")
    val text = content.plainText
    val tokens = LuceneTokenizer.tokenize(text, "english-lower")
    val luceneWordCount = tokens.length
    val stanDoc = StanfordAnnotator(text, "splitter")
    val stanfordSentenceCount = stanDoc.sentences().length
    (luceneWordCount, stanfordSentenceCount)
  }

  def newEnWiki() = {
    //English wiki
    val enConfig = new DatabaseConfiguration
    enConfig.setDatabase("kommlabir01fl")
    enConfig.setHost("kommlabir01fl")
    enConfig.setDatabase("wiki_english_20200401")
    enConfig.setUser("wundt")
    enConfig.setPassword("PaSSWoRD")
    enConfig.setLanguage(WikiConstants.Language.english)
    new Wikipedia(enConfig)
  }

  def newSiWiki() = {
    //Simple wiki
    val siConfig = new DatabaseConfiguration
    siConfig.setDatabase("kommlabir01fl")
    siConfig.setHost("kommlabir01fl")
    siConfig.setDatabase("wiki_simple_20200401")
    siConfig.setUser("wundt")
    siConfig.setPassword("PaSSWoRD")
    siConfig.setLanguage(WikiConstants.Language.simple_english)
    new Wikipedia(siConfig)
  }
}

case class PageInfo(id: Long, title: String, isDisambiguation: Boolean, isDiscussion: Boolean, redirectCount: Int, redirects: String, stubCount: Int, stubs: String, luceneWordCount: Int, stanfordSentenceCount: Int)
