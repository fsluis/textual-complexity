package ix.complexity.lucene3.esa.index

import java.io.File
import java.sql.DriverManager

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.lucene3.index.IndexReader
import org.apache.lucene3.store.FSDirectory

/**
 * Created by f on 08/09/16.
 */
/**
 * Creates a file with a serialized TIntIntHashMap that contains the number of inlinks pointing to each document
 * (wikipedia article), with the lucene docId (not the wikipedia pageId) as key.
 * @param index directory to write index
 * @param file the file to write to
 */
class EsaTitles(index: String, file: String) {
  // Lucene
  val indexDirectory = new File(index)
  val reader = IndexReader.open(FSDirectory.open(indexDirectory))

  // SQLite
  Class.forName("org.sqlite.JDBC")
  val database = DriverManager.getConnection("jdbc:sqlite:" + file)
  database.setAutoCommit(false)
  database.createStatement().executeUpdate("create table titles (id INTEGER, wikiId INTEGER, title TEXT);")
  val prep = database.prepareStatement("insert into titles values(?,?,?);")


  def run(): Unit = {
    val num = reader.numDocs()
    println("num: "+num)
    var found = 0
    var errors = 0
    for(i <- 0 until num) {

      if (!reader.isDeleted(i)) {
        try {
          // lucene
          val doc = reader.document(i)
          val wikiId = Integer.parseInt(doc.getField("id").stringValue())
          val title = doc.getField("title").stringValue()

          // sqlite
          prep.setInt(1, i)
          prep.setInt(2, wikiId)
          prep.setString(3, title)
          prep.addBatch()
          found+=1
        }
        catch {case _: Throwable => errors+=1}
        if(i%10000==0) {
          println("Found: "+found+", errors: "+errors)
          val updated = prep.executeBatch()
          println("Synced sqlite batch, "+updated.sum+" inserted")
        }
      }
      if (errors>1000) {
        println("More than 1000 errors, quiting")
        return
      }
    }
  }

  def close() {
    val updated = prep.executeBatch()
    println("Synced last sqlite batch, "+updated.sum+" inserted")
      reader.close()
    database.commit()
  }
}

object EsaTitles extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)

  override def main(args: Array[String]): Unit = {
    val index = args(0)
    val file = args(1)

    val indexer = new EsaTitles(index, file)
    try {
      indexer.run()
    } finally {
      indexer.close()
    }
  }
}