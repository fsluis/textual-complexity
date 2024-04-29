package ix.complexity.lucene3.esa

import java.sql.DriverManager

import gnu.trove.map.hash.TIntDoubleHashMap
import ix.common.util.TimeLogger
import org.apache.commons.logging.{LogFactory, Log}

/**
 * Created by f on 09/09/16.
 */
class ESA2Instance(file: String, alpha: Double = .5) {
  import ESA2Instance.timer

  // SQLite
  // Only single-threaded use a db-instance possible!
  Class.forName("org.sqlite.JDBC")
  val database = DriverManager.getConnection("jdbc:sqlite:" + file)

  // Only single-threaded use of these statements possible
  val links = database.prepareStatement("SELECT destId FROM left_join WHERE sourceId = ? AND numInLinksLog > ? " +
    "UNION SELECT destId FROM right_join WHERE sourceId = ? AND numInLinksLog > ?;")
  val inLinks = database.prepareStatement("SELECT numInLinksLog FROM concepts WHERE wikiId = ?")

  def toSecondOrder(in: TIntDoubleHashMap): TIntDoubleHashMap = {
    val out = new TIntDoubleHashMap(in)
    val it = out.iterator()
    while(it.hasNext) {
      val time = timer.start
      it.advance()
      val wikiId = it.key()
      val score = it.value()

      // Get number of inlinks log for this concept (i.e., the generality of this concept)
      inLinks.setInt(1, wikiId)
      val res1 = inLinks.executeQuery()
      val gen = if(res1.next()) res1.getDouble(1) else 0

      // Get the links that have a log-generality higher than gen+1
      links.setInt(1, wikiId)
      links.setDouble(2, gen+1)
      links.setInt(3, wikiId)
      links.setDouble(4, gen+1)
      val res2 = links.executeQuery()
      var sumLinkScore = 0d
      while(res2.next()) {
        val linkId = res2.getInt(1)
        sumLinkScore += in.get(linkId)
      }

      // Add sumLinkScore to this
      it.setValue(score+alpha*sumLinkScore)

      // Close & reuse the result sets
      res1.close()
      res2.close()
      timer.stop(time)
    }
    out
  }

  def close() = database.close()
}

object ESA2Instance {
  val log: Log = LogFactory.getLog(classOf[EsaModel])
  val timer: TimeLogger = new TimeLogger(log, "esa2", 100000)
}