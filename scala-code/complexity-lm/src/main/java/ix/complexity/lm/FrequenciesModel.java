package ix.complexity.lm;

import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.Multiset;
import ix.common.util.TimeLogger;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.sql.*;
import java.util.*;

import static java.lang.Math.log10;


/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 10/10/15
 * Time: 11:55 PM
 * To change this template use File | Settings | File Templates.
 */
public class FrequenciesModel {
    private static final Log log = LogFactory.getLog(FrequenciesModel.class);
    private static final TimeLogger tlog = new TimeLogger(log, "word_frequencies");
    private Connection database;
    private String databaseFile;
    public static int MAX_COLUMNS = 100;
    private String statementOne;
    private String statementAll;

    public FrequenciesModel(String databaseFile) {
        this.databaseFile = databaseFile;
    }

    // CREATE TABLE frequencies2 AS SELECT LOWER(word) as w, SUM(frequency) AS f FROM frequencies GROUP BY w COLLATE NOCASE;
    public void init() throws ClassNotFoundException, SQLException {
        // load the sqlite-JDBC driver using the current class loader
        Class.forName("org.sqlite.JDBC");
        database = DriverManager.getConnection("jdbc:sqlite:" + databaseFile);
        statementOne = "SELECT w, f FROM frequencies2 WHERE w=?";
        StringBuilder query = new StringBuilder("SELECT w,f FROM frequencies2 WHERE ");
        for(int i=0;i<MAX_COLUMNS; i++) {
            query.append("w=? ");
            if(i+1<MAX_COLUMNS)
                query.append("OR ");
        }
        statementAll = query.toString();
    }

    public List<Double> getFrequencies(Iterator<String> words) throws SQLException {
        long time = tlog.start();
        ImmutableMultiset<String> bag = ImmutableMultiset.copyOf(words);

        // Prepare query
        Set<String> subset = new HashSet<String>();
        Map<String, Long> frequencies = new HashMap<String,Long>();
        Iterator<String> iterator = bag.elementSet().iterator();
        int i=0;
        while(iterator.hasNext()) {
            if(i%MAX_COLUMNS==0) {
                if(subset.size()>0)
                    addFrequencies(subset, frequencies);
                subset.clear();
            }
            subset.add(iterator.next());
            i++;
        }
        addFrequencies(subset,frequencies);

        tlog.stop(time);

        LinkedList<Double> counts = new LinkedList<>();
        // Collect frequencies
        //int nan = 0;
        for(Multiset.Entry<String> entry : bag.entrySet()) {
            Long frequency = frequencies.get(entry.getElement());
            for(i=0; i<entry.getCount(); i++)
                if(frequency==null) {
                    counts.add(Double.NaN);
                    //nan++;
                } else
                    counts.add(log10(frequency));
        }
        return counts;
        //System.out.println(nan);
    }

    private void addFrequencies(Set<String> words, Map<String, Long> frequencies) throws SQLException {
        //System.out.println(query);
        if(words.size()==MAX_COLUMNS)
            addFrequenciesAll(words, frequencies);
        else
            for(String word : words)
                addFrequency(word, frequencies);
    }

    private void addFrequency(String word, Map<String, Long> frequencies) throws SQLException {
        PreparedStatement statement = database.prepareStatement(statementOne);
        statement.setString(1,word);
        //System.out.println(statementOne.toString());
        ResultSet rs = statement.executeQuery();
        if(rs.next())
            frequencies.put(rs.getString("w"), rs.getLong("f"));
        statement.close();
    }

    private void addFrequenciesAll(Set<String> words, Map<String, Long> frequencies) throws SQLException {
        PreparedStatement statement = database.prepareStatement(statementAll);
        int i=1;
        for(String word : words)
            statement.setString(i++,word);
        //System.out.println(statementAll.toString());
        ResultSet rs = statement.executeQuery();
        while(rs.next())
            frequencies.put(rs.getString("w"), rs.getLong("f"));
        statement.close();
    }

    public void setDatabaseFile(String databaseFile) {
        this.databaseFile = databaseFile;
    }

    public String getDatabaseFile() {
        return databaseFile;
    }

    public void close() throws SQLException {
        database.close();
    }
}
