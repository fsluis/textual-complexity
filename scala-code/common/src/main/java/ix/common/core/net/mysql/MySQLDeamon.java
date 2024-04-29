package ix.common.core.net.mysql;

import ix.common.core.net.Deamon;
import ix.common.core.net.DeamonException;
import ix.common.core.net.Server;
import ix.common.util.ConfigurationNode;

import java.util.Hashtable;
import java.util.Properties;

/**
 * This deamon handles MySQL support. Uses the MySQL driver as can been found in the lib dir.
 * @version 1.0
 */
public class MySQLDeamon implements Deamon {
    /**
     * The default name of this deamon, used for the getNickName() method implementation.
     */
    public static final String NAME = "MySQL";

    private Server parent;
    private int port = 3306;
    private Hashtable<String, MySQLDatabase> databases = new Hashtable<String, MySQLDatabase>();
    private Properties properties = new Properties();
    private boolean allowLoadFile = false;

    public MySQLDeamon() {
    }

    //set-queries
    void setPort(int port) {
        this.port = port;
    }

    void addDatabase(MySQLDatabase mySqlDatabase) {
        this.databases.put(mySqlDatabase.getName(), mySqlDatabase);
    }

    void setProperties(Properties properties) {
        this.properties = properties;
        String password = properties.getProperty("password");

    }

    void setAllowLoadFile(boolean allowLoadFile) {
        this.allowLoadFile = allowLoadFile;
    }

    //get-queries
    /**
     * Get the server to which this deamon belongs.
     * @return the hosting server
     */
    public Server getServer() {
        return parent;
    }

    /**
     * Returns the name of this class. See (@link ix.common.core.net.mysql.MySQLDeamon.NAME).
     * @return String containing the name.
     */
    public String getName() {
        return NAME;
    }

    /**
     * Get the database with the given name. If the database is unknown from the settings, it will be dynamically
     * added to the list of databases. This means it's not guaranteed that it will be available at all! For it to be
     * available it has to be created using the create database statement from mysql.
     * @param name the name of the database
     * @return the requested instance of MySQLDatabase
     */
    public MySQLDatabase getDatabase(String name) {
        if (databases.containsKey(name))
            return databases.get(name);

        MySQLDatabase database = new MySQLDatabase(this);
        database.setName(name);
        addDatabase(database);
        return database;
    }

    /**
     * The port of this deamon, default 3306.
     * @return the port of this deamon
     */
    public int getPort() {
        return port;
    }

    /**
     * The default properties for each connection of each database. Not public for security reasons,
     * keep it that way!
     * @return Properties instance.
     */
    Properties getProperties() {
        return properties;
    }

    /**
     * Returns true if this deamon supports the use of load file statement in it's queries.
     * @return true if load file is supported.
     */
    public boolean allowsLoadFile() {
        return allowLoadFile;
    }

    //methods
    /**
     * Configures this deamon. Please supply the following configure tuples:
     * data - the data classname, child of (@link ix.common.core.net.mysql.MySQLDeamonData).
     */
    public void configDeamon(Server parent, ConfigurationNode config) throws DeamonException {
        this.parent = parent;

        try {
            Class.forName("org.gjt.mm.mysql.Driver").newInstance();
            MySQLDeamonData data = new MySQLDeamonConfig();
            data.load(this, config);
        } catch(Exception e) {
            throw new DeamonException("error while configuring deamon", e);
        }
    }

    /**
     * Returns a MySQLDatabase instance with a connection to the deamon, but not for a specific database. Usefull
     * when no databases are available. Can only be used for the creation and dropping of databases.
     * @return MySQLDatabase instance with a connection but not for a specific database
     */
    public MySQLDatabase getNullaryDatabase() {
        return getDatabase("");
    }
}
