package ix.util.net2.jdbc;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 29-mrt-2010
 * Time: 18:28:51
 * To change this template use File | Settings | File Templates.
 */
public class Flushable {
    private PreparedStatement one, many;
    private int size = 4 * 2000; //lines per time
    private int values = 4;
    private List<Object> cache = Collections.synchronizedList(new LinkedList<Object>());

    public Flushable(Connection conn, String oneQ, int values, int size) throws SQLException {
        init(conn, oneQ, values, size, false, "");
    }

    public Flushable(Connection conn, String oneQ, int values, int size, boolean addPieceToOneQ) throws SQLException {
        init(conn, oneQ, values, size, addPieceToOneQ, "");
    }

    public Flushable(Connection conn, String oneQ, int values, int size, boolean addPieceToOneQ, String tail) throws SQLException {
        init(conn, oneQ, values, size, addPieceToOneQ, tail);
    }

    /**
     * Example use: flushable = new Flushable(getConnection(), "INSERT INTO links (id1,id2) VALUES", 2, 2000, true);         *
     * @param conn
     * @param oneQ Insert query
     * @param values
     * @param size
     * @param addPieceToOneQ Adds a " (?, ?, ?,)" piece to the one-query as well.
     * @param tail Adds a tail, eg ON CONFLICT("user", "contact") DO NOTHING
     * @throws SQLException
     */
    private void init(Connection conn, String oneQ, int values, int size, boolean addPieceToOneQ, String tail) throws SQLException {
        StringBuilder piece = new StringBuilder(" (?");
        for (int i = 0; i < values - 1; i++)
            piece.append(",?");
        piece.append(")");

        if(addPieceToOneQ)
            oneQ = oneQ.concat(piece.toString());

        StringBuilder manyQ = new StringBuilder(oneQ);
        for (int i = 0; i < size - 1; i++)
            manyQ.append(",").append(piece);

        manyQ.append(" ").append(tail);
        oneQ = oneQ + " " + tail;

        one = conn.prepareStatement(oneQ);
        many = conn.prepareStatement(manyQ.toString());
        this.values = values;
        this.size = values * size;
    }

    public void addDeliveries(Collection<? extends Object> values) {
        cache.addAll(values);
        checkFlush();
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
            System.err.println(many.toString());
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
                System.err.println(one.toString());
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
