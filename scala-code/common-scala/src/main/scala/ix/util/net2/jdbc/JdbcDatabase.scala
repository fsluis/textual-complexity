package ix.util.net2.jdbc

import java.sql.{DriverManager, Connection, PreparedStatement, Statement}
import java.util.Properties

import ix.util.net2.Servers
import ix.util.services2.ServiceException
import org.apache.commons.dbcp2.BasicDataSource
import org.apache.commons.logging.{LogFactory, Log}

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.reflect._
import scala.util.Try

/**
 * The database class, which handles the connection with the database.
 * @author f
 * @version 1.0
 */
abstract class JdbcDatabase(val name: String, val deamon: JdbcDaemon)  {
  val properties: Properties = new Properties
  for(entry <- deamon.properties.entrySet())
    properties.put(entry.getKey, entry.getValue)
  //properties.putAll(deamon.properties)

  // Initialize basic connection pool with default 3
  val pool = new BasicDataSource
  pool.setUrl(getUrl)
  pool.setUsername(properties.getProperty("user"))
  pool.setPassword(properties.getProperty("password"))
  properties.asScala.foreach{case (k,v) => pool.addConnectionProperty(k,v)}
  // make sure to set the correct driver at the mysql/postgres database!

  // Use one connection for the helper functions here
  // (which we don't return..., so it might not be best practice like this)
  private var localConnection: Connection = getPoolConnection

  def getUrl: String

  def isConnected: Boolean = Try {
    !localConnection.isClosed && localConnection.isValid(30)
  }.getOrElse(false)

  /**
   * Returns a connection from the pool. Make sure to close this connection after using it,
   * which returns the connection to the pool. Otherwise a deadlock will occur!
   * Use this one in a multi-threaded environment!
   * @return A connection
   */
  def getPoolConnection: Connection = pool.getConnection

  /**
   * Returns a shared connection used by this database class. Prefer to use getPoolConnection instead 
   * in a multi-threaded enviroment.
   * @return A connection
   */
  def connection: Connection = {
    if (!isConnected)
      localConnection = getPoolConnection
    localConnection
  }

  /**
   * Should return a brand new, single-use connection. For 'throw-away' use cases
   * such as with spark-sql-to-rdd (see Wikikids).
   * @return a brand-new connection
   */
  def getNewConnection: Connection = DriverManager.getConnection(getUrl, properties)

  def statement: Statement =
    localConnection.createStatement

  //def createStatement(): Statement = statement

  // Although it was intended to be private for security, it is quite handy to have this public for further use...
  // And usually when you can access this, you can also access the services.xml file already.
  //def getProperties = properties

  def query(query: String): PreparedStatement = {
    localConnection.prepareStatement(query)
  }
}

object JdbcDatabase {
  val log: Log = LogFactory.getLog(this.getClass)
  final val PostgresDatabaseTag = classTag[PostgresDatabase]
  final val MySQLDatabaseTag = classTag[MySQLDatabase]

  def apply[D <: JdbcDatabase](serverName: String, databaseName: String)
                              (implicit evd: ClassTag[D]): Option[D] = {

    val engineName = evd match {
      case MySQLDatabaseTag => MySQLDeamon.name
      case PostgresDatabaseTag => PostgresDaemon.name
      case _ => throw new ServiceException("Unknown database type requested: "+evd.runtimeClass+ "(available: "+PostgresDatabaseTag+","+MySQLDatabaseTag+")")
    }
    for {
      server <- Servers(serverName)
      daemon <- server.deamon[JdbcDaemon](engineName)
      database <- Option(daemon.database(databaseName).asInstanceOf[D])
    } yield database
  }
}
