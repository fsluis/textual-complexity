package ix.common.core.net.sql;

import java.sql.Connection;
import java.sql.Statement;

/**
 * Common database interface, which should be implemented by all sql-database classes.
 * 
 * @version 1.0
 */
public interface SQLDatabase {
    /**
     * Get the connection to the database. The connection returned is connected and working.
     * @return Connection instance
     * @throws SQLException if the connection failed
     */
    public Connection getConnection() throws SQLException;

    /**
     * Gives a new statement on which a query can be executed. May establish a connection if none is available,
     * and may give an exception if the connection fails.
     * @return Statement instance
     * @throws SQLException if any bad happens
     */
    public Statement createStatement() throws SQLException;
}
