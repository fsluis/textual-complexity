package ix.complexity.lucene3.esa.index

import java.io.File
import java.sql.DriverManager

import ix.data.wikipedia.WikipediaService
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.lucene3.index.{Term, TermDocs}
import org.apache.lucene3.search.IndexSearcher
import org.apache.lucene3.store.{Directory, FSDirectory}

import scala.collection.JavaConversions._
import scala.collection.mutable

/**
 * Creates a sqlite that contains the number of inlinks pointing to each document, as well as the links between docs.
 * @param wiki wikipedia version to use (see WikipediaService)
 * @param index directory to read index
 * @param file the db file to write to
 */
class Esa2Indexer(wiki: String, index: String, file: String) {
  import Esa2Indexer.log

  // Lucene
  val dir: Directory = FSDirectory.open(new File(index))
  val searcher: IndexSearcher = new IndexSearcher(dir, true)
  val reader = searcher.getIndexReader

  // SQLite
  Class.forName("org.sqlite.JDBC")
  val database = DriverManager.getConnection("jdbc:sqlite:" + file)
  database.setAutoCommit(false)
  database.createStatement().executeUpdate("create table concepts (docId INTEGER UNIQUE, wikiId INTEGER UNIQUE, title TEXT, numOfInLinks INT, numInLinksLog DOUBLE);")
  val concept = database.prepareStatement("insert into concepts values(?,?,?,?,?);")
  database.createStatement().executeUpdate("create table links (sourceId INTEGER, destId INTEGER);")
  val link = database.prepareStatement("insert into links values(?,?);")

  def run() {
    val articles = WikipediaService.iterator(wiki, 10000)
    var nrOfConcepts = 0
    var nrOfLinks = 0
    var totalLinks: Long = 0
    var errors = 0
    for{
      page <- articles
      docId: Int <- queryDocId(page.getPageId)
    } try {
      // wiki
      val wikiId = page.getPageId
      val title = page.getTitle.toString
      val inIds = page.getInlinkIDs
      //val outIds = page.getOutlinkIDs
      val ids = inIds //++ outIds
      val nrOfInLinks = inIds.size()
      //val nrOfOutLinks = outIds.size()
      val nrOfInLinksLog = math.log10(nrOfInLinks)
      //val nrOfOutLinksLog = math.log10(nrOfOutLinks)

      // sqlite
      concept.setInt(1, docId)
      concept.setInt(2, wikiId)
      concept.setString(3, title)
      concept.setInt(4, nrOfInLinks)
      //concept.setInt(5, nrOfOutLinks)
      concept.setDouble(5, nrOfInLinksLog)
      //concept.setDouble(7, nrOfOutLinksLog)
      concept.addBatch()
      nrOfConcepts+=1

      // sqlite
      for(id <- ids) {
        link.setInt(1, wikiId)
        link.setInt(2, id)
        link.addBatch()
        nrOfLinks+=1
      }

    } catch {
      case e: Throwable =>
        log.warn("Error while processing concept", e)
        errors+=1
    } finally {
      // sync
      if(nrOfLinks>100000) {
        val updatedLinks = link.executeBatch().sum
        print("\r")
        totalLinks += nrOfLinks
        log.info(s"Synced $updatedLinks (total $totalLinks) links. Errors: "+errors)
        database.commit()
        nrOfLinks=0
      }
      if(nrOfConcepts%1000==0) {
        val updatedConcepts = concept.executeBatch().sum
        print("\r")
        log.info(s"Synced $updatedConcepts (total $nrOfConcepts) concepts. Errors: "+errors)
        database.commit()
      }

      // Test break
      //if(nrOfConcepts%1000==0) return

      print(s"\rConcepts: $nrOfConcepts, links: $nrOfLinks")
    }
  }

  def queryDocId(wikiId: Long): Option[Int] = {
    val term: Term = new Term("id", wikiId.toString)
    val termDocs: TermDocs = reader.termDocs(term)
    val docIds = new mutable.HashSet[Int]
    while (termDocs.next) docIds += termDocs.doc
    if (docIds.size == 1)
      Some(docIds.head)
    else {
      print("\r")
      log.warn("Wiki id of " + wikiId + " found " + docIds.size + " times in index")
      None
    }
  }

    def close(): Unit = {
      print("\r")
      log.info(s"Closing index reader")
      reader.close()
      val updatedConcepts = concept.executeBatch().sum
      val updatedLinks = link.executeBatch().sum
      log.info(s"Synced last sqlite batch of $updatedConcepts concepts and $updatedLinks links")
      log.info(s"Creating indices")
      database.createStatement().executeUpdate("CREATE INDEX source_ids on links (sourceId);")
      database.createStatement().executeUpdate("CREATE INDEX dest_ids on links (destId);")
      log.info(s"Creating view table")
      database.createStatement().executeUpdate("" +
        "CREATE VIEW left_join AS " +
        "SELECT links.sourceId, links.destId, concepts.numInLinksLog " +
        "FROM links LEFT OUTER JOIN concepts ON links.destId=concepts.wikiId;")
      database.createStatement().executeUpdate("" +
        "CREATE VIEW right_join AS " +
        "SELECT links.destId AS sourceId, links.sourceId AS destId, concepts.numInLinksLog " +
        "FROM links LEFT OUTER JOIN concepts ON links.sourceId=concepts.wikiId;")
      database.commit()
    }
}

object Esa2Indexer extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  override def main(args: Array[String]): Unit = {
    val wiki = args(0)
    val index = args(1)
    val file = args(2)

    val indexer = new Esa2Indexer(wiki,index,file)
    indexer.run()
    indexer.close()
  }
}