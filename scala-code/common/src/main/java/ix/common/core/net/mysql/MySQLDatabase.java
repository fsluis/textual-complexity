package ix.common.core.net.mysql;

import com.mysql.cj.jdbc.MysqlDataSource;
import ix.common.core.net.sql.SQLDatabase;
import ix.common.core.net.sql.SQLException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.Statement;
import java.util.Enumeration;
import java.util.Properties;

/**
 * The database class, which handles the connection with the database.
 * @version 1.0
 */
public class MySQLDatabase implements SQLDatabase {
	private String name;
	private Connection connection;
	private Properties properties;
	private MySQLDeamon parent;
    private MysqlDataSource source;
    private static final Log log = LogFactory.getLog(MySQLDatabase.class);

	//constructors
	/**
     * Constructor. Sets the parent deamon and the default properties of this deamon for the connection.
     * @param parent the parent deamon
     */
    MySQLDatabase(MySQLDeamon parent) {
        this.parent = parent;
        properties = new Properties(parent.getProperties());
	}

	//set-queries
	void setName(String name) {
		this.name = name;
	}

	void setProperties(Properties properties) {
		this.properties.putAll(properties);
	}

	//get-queries
	/**
     * SQLDatabase name.
     * @return String
     */
    public String getName() {
		return name;
	}

	/**
     * Parent deamon of this database.
     * @return MySQLDeamon
     */
    public MySQLDeamon getDeamon() {
		return parent;
	}

	//methods
    /**
     * Gives a new statement on which a query can be executed. May establish a connection if none is available,
     * and may give an exception if the connection fails.
     * @return Statement instance
     * @throws SQLException if any bad happens
     */
    public Statement createStatement() throws SQLException {
        try {
            return getConnection().createStatement();
        } catch (java.sql.SQLException e) {
            throw new SQLException(e);
        }
    }

    private Connection connect() throws SQLException {
		try {
			//return DriverManager.getConnection("jdbc:mysql://" + parent.getServer().getHost() + ":" + parent.getPort() + "/" + name, properties);
            return getSource().getConnection();
        } catch(java.sql.SQLException e) {
			throw new SQLException("Unable to connect to server jdbc:mysql://" + parent.getServer().getHost() + ":" + parent.getPort() + "/" + name + " (user: "+properties.getProperty("user")+")", e);
		}
	}

	private boolean isConnected() {
		if (connection != null)
			try {
				if ( !connection.isClosed() && connection.isValid(30))
					return true;
			} catch(java.sql.SQLException e) {
				return false;
			}
		return false;
	}

	public Connection getConnection() throws SQLException {
		if (!isConnected())
			connection = connect();
		return connection;
	}

    public Connection getNewConnection() throws SQLException {
        return connect();
    }

    public DataSource getSource() throws SQLException {
        if(source==null) {
            source = new MysqlDataSource();
            source.setServerName(parent.getServer().getHost());
            source.setPort(parent.getPort());
            source.setDatabaseName(name);

            Enumeration<?> iterator = properties.propertyNames();
            while(iterator.hasMoreElements()) {
                String name = iterator.nextElement().toString();
                if("user".equals(name))
                    source.setUser(properties.getProperty(name));
                else if("password".equals(name))
                    source.setPassword(properties.getProperty(name));
                else
                    log.warn("Unknown property type: "+name);
            }
        }

        return source;
    }
}
