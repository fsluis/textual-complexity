package ix.data.com.datasets

import java.util

import ix.data.com.datasets.WikiPairsTable.sidb
import ix.data.hadoop.HadoopRecord
import ix.data.structure.{Content, ContentList, DataSet}
import ix.common.data.wikipedia.BuildTables.PageRecord
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}
import org.apache.hadoop.conf.{Configuration => HConfiguration}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{LongWritable, SequenceFile}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.ExtKeyFunctions.ext
import org.apache.spark.rdd.ExtKeyFunctions._
import org.apache.spark.rdd.{ExtKey, ExtKeyTracker, RDD}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.execution.datasources.jdbc.JDBCRDD

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.collection.immutable.ListMap
import scala.util.matching.Regex

/**
 * DataSet access to EnSiWiki sequence files
 */
class EnSiWikiDataSet extends DataSet {
  val EnSiWikiRE: Regex = "ensiwiki_(\\d{4})-([a-z/_]+)-([a-z0-9]+)".r

  var filename: String = System.getProperty("user.home") + "resources/data/ensiwiki_2020-ids_10.hsf"
  var size: Option[String] = Some("10")
  var name: String = ""
  var year: String = "2020"
  var version: String = ""

  override def init(name: String, conf: Configuration): Unit = {
    filename = conf.getString("filename", filename)
    val EnSiWikiRE(year, version, size) = name
    this.year = year
    this.version = version
    this.size = Some(size)
    //size = Option(conf.getString("size", "")).filter(_.nonEmpty)
    this.name = name
  }

  def loadData(sc: SparkContext): RDD[(LongWritable, EnSiWikiPageRecord)] = {
    // Load pages, add keys (page, lang)
    val minPartitions = size.map( _.replaceAll("k", "000") ).map( _.toInt / 50 ).map( math.max(_, 10) ).getOrElse(100)
    // minimal partitions: 10 < size/50 with default 100
    // thus, 50 texts per partition
    sc.sequenceFile(filename, classOf[LongWritable], classOf[EnSiWikiPageRecord], minPartitions)
  }

  override def load(sc: SparkContext)(implicit tracker: ExtKeyTracker): (ListMap[String, RDD[_]], ListMap[String, RDD[(ExtKey[Long], Content)]]) = {
    val data = loadData(sc)
    val pages = ext(data.map{case (k,v) => (k.get, v)})
    val pairs = pages.kvMap{case (id, record) => record.getLong("pair_id") -> record }
    //val sorted = if(sortByPairId) pairs.sortBy(_._1._1) else pairs
    val lang = pairs.kvMap{case (id, record) => record.getString("lang") -> record }

    // Parse, add keys (wiki_id, section_id, paragraph_id)
    import ix.data.wikipedia.WikipediaParser
    val art: RDD[(ExtKey[Long], Content)] = lang.kvMap{ case (k, v) => (v.getLong("wiki_id"), WikipediaParser(v.getString("wiki_text"), "flush")) }
    val sec: RDD[(ExtKey[Long], Content)] = art.kvFlatMap( _._2.asInstanceOf[ContentList].sorted.map(c => c.charIndex -> c) )
    val par: RDD[(ExtKey[Long], Content)] = sec.kvFlatMap( _._2.asInstanceOf[ContentList].sorted.map(c => c.charIndex -> c) )

    //: Map[String, RDD[(Long, Content)]]
    val contentRDDs = ListMap("article" -> art, "section" -> sec, "paragraph" -> par)
    val keyRDDs: ListMap[String, RDD[_]] = ListMap("page_id" -> pages, "pair_id" -> pairs, "lang" -> lang)
    (keyRDDs, contentRDDs)
  }
}


/**
 * Object/APP to export data from EnSiWiki DB to EnSiWiki HSF (hadoop sequence files)
 */
object EnSiWikiDataSet extends App {
  lazy val log: Log = LogFactory.getLog(this.getClass)
  lazy val sqlUrl = "jdbc:mysql://kommlabir01fl/ensiwiki_2020"
  lazy val ensidb = ix.util.net2.jdbc.MySQLDatabase("kommlabir01fl", "ensiwiki_2020").get

  /*def stats(query: String) = {
    val stmnt = ensidb.connection.prepareStatement("SELECT MIN(id), MAX(id), COUNT(*) FROM ("+query+") AS data")
    val res = stmnt.executeQuery()
    res.next()
    val (min, max, count) = (res.getLong(1), res.getLong(2), res.getLong(3))
    (min,max,count)
  }



  def test(sc: SparkContext, sql: SQLContext) = {
    val query = "SELECT * FROM (SELECT * FROM pages LIMIT 0,20) AS pages WHERE ? <= id and id <= ?"
    // first ? is lowerbound, second is upperbound
    // as long as lowerbound is always zero, the second param to limit can be interpreted as upper bound
    val rdd = new org.apache.spark.rdd.JdbcRDD(
      sc,
      () => { ix.util.net2.jdbc.MySQLDatabase("kommlabir01fl", "ensiwiki_2020").get.getNewConnection },
      sql = query,
      0, 10, 2)

    // ORDER BY agerank_simple
    // SELECT MIN(agerank), MAX(agerank) FROM pairs_ranking;
    val pages = sql.read.jdbc(sqlUrl,
      "(SELECT id, lang, page_id, title, text FROM pages ORDER BY agerank) AS pages",
      columnName = "agerank",
      lowerBound = 0,
      upperBound = 7000,
      numPartitions = 100,
      ix.data.com.COM1.sqlOptions)
//    val convertor = Encoders.bean(classOf[ix.data.com.datasets.EnSiWikiPage])
//    val rdd = pages.as[ix.data.com.datasets.EnSiWikiPage](convertor).rdd
    rdd.getNumPartitions
  } */


  override def main(args: Array[String]): Unit = {
    val config = args(0)
    val size = args(1)
    val columns = "id,lang,page_id,pairs_id,title,text"
    val year = "2020"

    val printSize = if (size.toInt>=1000) s"${size.toInt/1000}k" else size

    val (filename, query) = config match {
      case "id" => (s"ensiwiki_$year-id-$printSize.hsf", s"SELECT $columns FROM pages WHERE pairs_disambiguation=0 ORDER BY id LIMIT 0,$size")
      case "agerank" => (s"ensiwiki_$year-agerank-$printSize.hsf", s"SELECT $columns FROM pages WHERE pairs_disambiguation=0 ORDER BY agerank LIMIT 0,$size")
      case "agerank_nostub" => (s"ensiwiki_$year-agerank_nostub-$printSize.hsf", s"SELECT $columns FROM pages WHERE pairs_disambiguation=0 AND pairs_stub_count=0 ORDER BY agerank LIMIT 0,$size")
      case "simpleagerank" => (s"ensiwiki_$year-simpleagerank-$printSize.hsf", s"SELECT $columns FROM pages WHERE pairs_disambiguation=0 ORDER BY simple_agerank LIMIT 0,$size")
      case "simpleagerank_nostub" => (s"ensiwiki_$year-simpleagerank_nostub-$printSize.hsf", s"SELECT $columns FROM pages WHERE pairs_disambiguation=0 AND pairs_stub_count=0 ORDER BY simple_agerank LIMIT 0,$size")
    }
    createSequenceFile(query,filename)
  }

  /**
   * Converting to SL has three main advantages:
   * 1. Can use it everywhere
   * 2. Static dataset
   * 3. Spark/Hadoop can handle parallelization easily
   *
   * @param query
   * @param filename
   */
  def createSequenceFile(query: String, filename: String): Unit = {

    // Open writer
    val conf: HConfiguration = new HConfiguration
    val fs: FileSystem = FileSystem.get(conf)
    val writer = SequenceFile.createWriter(fs, conf, new Path(filename), classOf[LongWritable], classOf[EnSiWikiPageRecord])

    // Get database
    val database = ix.util.net2.jdbc.MySQLDatabase("kommlabir01fl", "ensiwiki_2020").get
    val statement = database.connection.prepareStatement(query)
    log.info("Running query: "+query)
    val res = statement.executeQuery()

    log.info("Writing to file: "+filename)
    // Transfer data
    while(res.next()) {
      val rec = new EnSiWikiPageRecord()
      //val agerank = res.getDouble("agerank")
      val id = res.getLong("id")
      rec.put("page_id", id) // primary id of Pages table
      rec.put("lang", res.getString("lang"))
      rec.put("wiki_id", res.getLong("page_id"))  // primary id of Page table in respective Wikis
      rec.put("pair_id", res.getLong("pairs_id"))  // primary id of pairs table
      rec.put("title", res.getString("title"))
      rec.put("wiki_text", res.getString("text"))
      writer.append(new LongWritable(id), rec)
    }

    // Close
    // I doubt whether this sync-command is necessary, doesn't seem to change the final file size
    // don't use the sync() command though, as that one adds a syncpoint to the file (rather than flushing it)
    //writer.syncFs()
    // "Flush out the data in client's user buffer. After the return of this call, new readers will see the data." (Syncable.java)
    writer.hflush()
    //writer.hsync()
    writer.close()

    // bye bye
    log.info("Done")
  }
}

//case class EnSiWikiPage(agerank: Double, lang: String, page_id: Long, title: String, wiki_text: String)

import HadoopRecord._
class EnSiWikiPageRecord extends HadoopRecord {
  val types = Map("page_id" -> LONG, "lang" -> STRING, "wiki_id" -> LONG, "pair_id" -> LONG, "title" -> STRING, "wiki_text" -> STRING)
  override def getTypes: util.Map[String, Integer] = types.mapValues(new Integer(_))
}