package ix.common.core.net.mysql;

import ix.common.core.net.NetException;
import ix.common.core.net.Server;
import ix.common.core.net.ServerService;
import ix.common.core.services.ServiceException;
import ix.common.core.services.Services;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Types;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 20-Dec-2010
 * Time: 15:26:23
 * To change this template use File | Settings | File Templates.
 */
public class MySQLHelper {
    private static final Log log = LogFactory.getLog(MySQLHelper.class);
    private MySQLDatabase database;

    public MySQLHelper(MySQLDatabase database) {
        this.database = database;
    }

    public MySQLDatabase getDatabase() {
        return database;
    }

    public Connection getConnection() throws ix.common.core.net.sql.SQLException {
        return database.getConnection();
    }

    public static void setNumber(PreparedStatement statement, int index, Number number) throws SQLException {
        if (number instanceof Integer)
            statement.setInt(index, number.intValue());
        else if (number instanceof Float)
            if (Float.isNaN(number.floatValue()))
                statement.setNull(index, Types.FLOAT);
            else
                statement.setFloat(index, number.floatValue());
        else if (number instanceof Double)
            if (Double.isNaN(number.doubleValue()))
                statement.setNull(index, Types.DOUBLE);
            else
                statement.setDouble(index, number.doubleValue());
        else
            throw new SQLException("Could not determine number type of: " + number);
    }

    public static MySQLDatabase loadLocalDatabase(String databaseName) throws NetException, ServiceException {
        return loadDatabase(null, databaseName);
    }

    public static MySQLDatabase loadDatabase(String serverName, String databaseName) throws NetException, ServiceException {
        ServerService servers = ((ServerService) Services.getDefault().getService(ServerService.class));
        Server server;
        if(serverName == null)
            server = servers.getLocalServer();
        else
            server = servers.getServer(serverName);
        log.info("Serving to mysql database "+databaseName+" on server "+server.getName());
        MySQLDeamon deamon = (MySQLDeamon) server.getDeamon(MySQLDeamon.NAME);
        return deamon.getDatabase(databaseName);
    }

    /**
     * Created by IntelliJ IDEA.
     * User: Frans
     * Date: 29-mrt-2010
     * Time: 18:28:51
     * To change this template use File | Settings | File Templates.
     */
    public static class Flushable {
        private PreparedStatement one, many;
        private int size = 4 * 2000; //lines per time
        private int values = 4;
        private List<Object> cache = Collections.synchronizedList(new LinkedList<Object>());

        public Flushable(Connection conn, String oneQ, int values, int size) throws SQLException {
            init(conn, oneQ, values, size, false);
        }

        /**
         * Example use: flushable = new Flushable(getConnection(), "INSERT INTO links (id1,id2) VALUES", 2, 2000, true);         *
         * @param conn
         * @param oneQ
         * @param values
         * @param size
         * @param addPieceToOneQ
         * @throws SQLException
         */
        public Flushable(Connection conn, String oneQ, int values, int size, boolean addPieceToOneQ) throws SQLException {
            init(conn, oneQ, values, size, addPieceToOneQ);
        }

        private void init(Connection conn, String oneQ, int values, int size, boolean addPieceToOneQ) throws SQLException {
            StringBuilder piece = new StringBuilder(" (?");
            for (int i = 0; i < values - 1; i++)
                piece.append(",?");
            piece.append(")");

            if(addPieceToOneQ)
                oneQ = oneQ.concat(piece.toString());

            StringBuilder manyQ = new StringBuilder(oneQ);
            for (int i = 0; i < size - 1; i++)
                manyQ.append(",").append(piece);

            one = conn.prepareStatement(oneQ);
            many = conn.prepareStatement(manyQ.toString());
            this.values = values;
            this.size = values * size;
        }

        public void addLine(Collection<? extends Object> values) {
            cache.addAll(values);
            checkFlush();
        }

        public void finish() {
            flushOne();
        }

        public void checkFlush() {
            if (cache.size() == size)
                flush();
        }

        private void flush() {
            if (cache.size() == size) {
                flushMany();
            } else {
                flushOne();
            }
        }

        private void flushMany() {
            try {
                for (int i = 0; i < cache.size(); i++) {
                    many.setObject(i + 1, cache.get(i));
                }
                many.execute();
            } catch (SQLException e) {
                System.out.println(many.toString());
                e.printStackTrace();
                flushOne();     //if the big batch fails, do them one by one
            }
            cache.clear();
        }

        private void flushOne() {
            for (int i = 0; i < cache.size(); i = i + values) {
                try {
                    for (int j = 0; j < values; j++) {
                        one.setObject(j + 1, cache.get(i + j));
                    }
                    one.execute();
                } catch (SQLException e) {
                    System.out.println(one.toString());
                    e.printStackTrace();
                }
            }
            cache.clear();
        }

        public void addLine(Object... values) {
            try{
                for(Object value : values)
                    addValue(value);
                checkFlush();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        public void addValue(Object value) {
            if(value instanceof Double)
                if(((Double) value).isInfinite())
                    value = null;
            cache.add(value);
        }
    }
}
