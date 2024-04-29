package ix.complexity.lucene3.esa.index

import java.io.File
import java.util.logging.{Level, Logger}

import de.tudarmstadt.ukp.wikipedia.parser.mediawiki.MediaWikiParserFactory
import de.tudarmstadt.ukp.wikipedia.parser.{ParsedPage, Template}
import ix.complexity.lucene3.tokenizer.Tokenizer
import ix.data.wikipedia.WikipediaService
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.lucene3.document.{Document, Field}
import org.apache.lucene3.index.IndexWriter
import org.apache.lucene3.store.{FSDirectory, SimpleFSLockFactory}

import scala.collection.JavaConversions._

/**
 * Creates an ESA index. The documents (docId) are wikipedia articles.
 * @param wiki wikipedia version to use (see WikipediaService)
 * @param index directory to write index
 * @param lockDir /tmp dir
 */
class EsaIndexer(wiki: String, index: String, tokenizer: String, lockDir: String = "/tmp") {

  import EsaIndexer.log

  log.info("Loading wiki " + wiki)

  val articles = WikipediaService.iterator(wiki, 10000)
  val analyzer = Tokenizer.get(tokenizer)
  val writer = new IndexWriter(FSDirectory.open(new File(index), new SimpleFSLockFactory(lockDir)), analyzer, IndexWriter.MaxFieldLength.UNLIMITED)
  writer.setUseCompoundFile(true)
  val parser = (new MediaWikiParserFactory).createParser

  def run() {
    log.info("Starting run, dots per 10,000 articles")
    var i = 0
    for (page <- articles)
      if (!page.isDisambiguation && !page.isRedirect) try {
        val parsed = parser.parse(page.getText)
        if (isStub(parsed).isEmpty) try {
          // Index
          val doc: Document = new Document
          doc.add(new Field("id", page.getPageId.toString, Field.Store.YES, Field.Index.NOT_ANALYZED))
          doc.add(new Field("title", page.getTitle.toString, Field.Store.YES, Field.Index.ANALYZED))
          doc.add(new Field("content", page.getPlainText, Field.Store.NO, Field.Index.ANALYZED))
          writer.addDocument(doc)

          // Log
          i += 1
          if (i % 10000 == 0) print(".")
        } catch {
          case e: Throwable => log.warn("Unable to add page " + i + ", wiki pageid " + page.getPageId, e)
        }
      } catch {
        case e: Throwable => log.warn("Unable to parse page " + i)
      }
    println("done")
    log.info("Done with run")
  }

  def close() {
    writer.optimize
    writer.close()
    log.info("Closed index")
  }

  private def isStub(parsed: ParsedPage): Option[String] = {
    parsed.getTemplates.
      map((template: Template) => template.getName).
      find((name: String) => name.contains("stub"))
  }
}

object EsaIndexer extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  override def main(args: Array[String]): Unit = {
    System.setProperty("org.apache.lucene.commitLockTimeout", "60000")
    System.setProperty("org.apache.lucene.writeLockTimeout", "60000")
    System.setProperty("org.apache.lucene.lockdir", "/tmp")
    Logger.getLogger("de.tudarmstadt.ukp.wikipedia").setLevel(Level.OFF)

    val wiki = args(0)
    val index = args(1)
    val tokenizer = args(2)

    val indexer = new EsaIndexer(wiki, index, tokenizer)
    try {
      indexer.run()
    }
    finally {
      indexer.close()
    }
  }
}