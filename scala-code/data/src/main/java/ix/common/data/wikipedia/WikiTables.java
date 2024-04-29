package ix.common.data.wikipedia;

import ix.data.hadoop.HadoopRecord;
import ix.util.net2.jdbc.MySQLDatabase;

import java.sql.*;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by f on 10/03/16.
 */
public class WikiTables {
    public static class PageRecord extends HadoopRecord {
        public static final Map<String, Integer> TYPES = new LinkedHashMap<String, Integer>() {{
            put("id", LONG);
            put("lang", STRING);
            put("wiki_id", INT);
            put("stub", STRING);
            put("disambiguation", BOOL);
            put("redirect", BOOL);
            put("wiki_text", TEXT);
        }};

        @Override
        public Map<String, Integer> getTypes() {
            return TYPES;
        }
    }

    public static class PagesTable {
        private PreparedStatement insert;
        private MySQLDatabase database;
        private String name;

        public Connection getConnection() {
            return database.connection();
        }

        public PagesTable(MySQLDatabase database, String name) throws SQLException {
            this.database = database;
            this.name = name;

            insert = getConnection().prepareStatement("INSERT INTO " + name + " " +
                    "(lang,wiki_id,title,stub,is_disambiguation,is_redirect,length,first_revision,last_revision,number_of_revisions,wiki_text) " +
                    "VALUES " +
                    "(?,?,?,?,?,?,?,?,?,?,?)", PreparedStatement.RETURN_GENERATED_KEYS);
        }

        public boolean create() throws SQLException {
            String query = "CREATE TABLE IF NOT EXISTS "+name+" (" +
                    "id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, " +
                    "lang VARCHAR(25) NOT NULL, " +
                    "wiki_id BIGINT UNSIGNED NOT NULL, " +
                    "title VARCHAR(255) NULL," +
                    "stub VARCHAR(255) NULL, " +
                    "is_disambiguation tinyint(2) default 0, " +
                    "is_redirect tinyint(2) default 0, " +
                    "length INT UNSIGNED NOT NULL default 0, " +
                    "first_revision DATETIME NULL, " +
                    "last_revision DATETIME NULL, " +
                    "number_of_revisions INT UNSIGNED NOT NULL DEFAULT 0, " +
                    "wiki_text LONGTEXT NOT NULL," +
                    "UNIQUE KEY `wiki_id` (`wiki_id`)" +
                    ") ENGINE = MYISAM DEFAULT CHARSET=utf8;";
            Statement statement = getConnection().createStatement();
            return statement.execute(query);
        }

        public long addPage(String language, int wikiId, String title, String stub, boolean disambiguation, boolean redirect, int length, Date firstRevision, Date lastRevision, int numberOfRevisions, String text) throws SQLException {
            int i = 0;
            insert.setString(++i, language);
            insert.setLong(++i, wikiId);
            insert.setString(++i, title);
            insert.setString(++i, stub);
            insert.setBoolean(++i, disambiguation);
            insert.setBoolean(++i, redirect);
            insert.setInt(++i, length);
            if(firstRevision!=null)
                insert.setDate(++i, new java.sql.Date(firstRevision.getTime()));
            else
                insert.setDate(++i, null);
            if(lastRevision!=null)
                insert.setDate(++i, new java.sql.Date(lastRevision.getTime()));
            else
                insert.setDate(++i, null);
            insert.setInt(++i, numberOfRevisions);
            insert.setString(++i, text);
            insert.execute();
            ResultSet result = insert.getGeneratedKeys();
            if (result.next())
                return result.getLong(1);
            return -1;
        }
    }
}
