package ix.common.core.net.sql;

/**
 * Throwed when an exception occurs in a sql deamon package.
 *
 * @version 1.0
 */
public class SQLException extends java.sql.SQLException {
    private final static String message = "Exception in an SQL Deamon";

    public SQLException() {
        super(message);
    }

    public SQLException(String message) {
        super(message);
    }

    public SQLException(String message, Exception cause) {
        super(message, cause);
    }

    public SQLException(Exception cause) {
        super(message, cause);
    }
}
