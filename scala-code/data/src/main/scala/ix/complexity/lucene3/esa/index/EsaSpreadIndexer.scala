package ix.complexity.lucene3.esa.index

import java.io.File

import gnu.trove.map.hash.TIntIntHashMap
import ix.common.util.Utils
import ix.data.wikipedia.WikipediaService
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.lucene3.index.{Term, TermDocs}
import org.apache.lucene3.search.IndexSearcher
import org.apache.lucene3.store.{Directory, FSDirectory}

import scala.collection.mutable

/**
 * Creates a file with a serialized TIntIntHashMap that contains the number of inlinks pointing to each document
 * (wikipedia article), with the lucene docId (not the wikipedia pageId) as key.
 * @param wiki wikipedia version to use (see WikipediaService)
 * @param index directory to write index
 * @param file the file to write to
 */
class EsaSpreadIndexer(wiki: String, index: String, file: String) {
  val articles = WikipediaService.iterator(wiki, 10000)
  val dir: Directory = FSDirectory.open(new File(index))
  val searcher: IndexSearcher = new IndexSearcher(dir, true)
  val reader = searcher.getIndexReader
  val generality: TIntIntHashMap = new TIntIntHashMap

  def run() {
    for{
      page <- articles
      docId: Int <- docId(page.getPageId)
    } {
      val nrOfInLinks: Int = page.getNumberOfInlinks
      generality.put(docId, nrOfInLinks)
    }
  }

  def docId(wikiId: Long): Option[Int] = {
    val term: Term = new Term("id", wikiId.toString)
    val termDocs: TermDocs = reader.termDocs(term)
    val docIds = new mutable.HashSet[Int]
    while (termDocs.next) docIds += termDocs.doc
    if(docIds.size==1)
      Some(docIds.head)
    else {
      EsaTitles.log.warn("Wiki id of "+wikiId+" found "+docIds.size+" times in index")
      None
    }
  }

  def close() {
    Utils.writeToFile(file, generality)
  }
}

object EsaSpreadIndexer extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  override def main(args: Array[String]): Unit = {
    val wiki = args(0)
    val index = args(1)
    val file = args(2)

    val indexer = new EsaSpreadIndexer(wiki,index,file)
    indexer.run()
    indexer.close()
  }
}