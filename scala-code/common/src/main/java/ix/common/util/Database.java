package ix.common.util;

import java.sql.*;
import java.util.List;
import java.util.LinkedList;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 1-apr-2010
 * Time: 17:49:11
 * To change this template use File | Settings | File Templates.
 */
public abstract class Database {
    private static String jdbc_url = "jdbc:mysql://130.89.13.112/complexity";
    private static String jdbc_username = "root";
    private static String jdbc_password = "Myklemie";
    private static String jdbc_driver = "org.gjt.mm.mysql.Driver";
    private Connection connection;

    public Connection getNewConnection() throws SQLException {
        try {
            Class.forName(jdbc_driver);
        } catch (ClassNotFoundException e) {
            throw new SQLException(e);
        }
        return DriverManager.getConnection(jdbc_url, jdbc_username, jdbc_password);
    }

    public Connection getConnection() throws SQLException {
        if (connection != null)
            if ( !connection.isClosed() && connection.isValid(30))
                return connection;

        connection = getNewConnection();

        return connection;
    }

    public void init() throws SQLException {
        Connection conn = getConnection();
    }

    public void exit() {
    }

}
