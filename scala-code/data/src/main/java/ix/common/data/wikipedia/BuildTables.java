package ix.common.data.wikipedia;

import de.tudarmstadt.ukp.wikipedia.api.DatabaseConfiguration;
import de.tudarmstadt.ukp.wikipedia.api.Page;
import de.tudarmstadt.ukp.wikipedia.api.WikiConstants;
import de.tudarmstadt.ukp.wikipedia.api.Wikipedia;
import de.tudarmstadt.ukp.wikipedia.api.exception.WikiApiException;
import de.tudarmstadt.ukp.wikipedia.api.exception.WikiInitializationException;
import de.tudarmstadt.ukp.wikipedia.parser.Link;
import de.tudarmstadt.ukp.wikipedia.parser.ParsedPage;
import de.tudarmstadt.ukp.wikipedia.parser.Template;
import de.tudarmstadt.ukp.wikipedia.parser.mediawiki.MediaWikiParser;
import de.tudarmstadt.ukp.wikipedia.parser.mediawiki.MediaWikiParserFactory;
import ix.common.core.net.NetException;
import ix.data.hadoop.HadoopRecord;
import ix.util.net2.jdbc.MySQLDatabase;
import ix.common.core.services.ServiceException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;

import java.io.IOException;
import java.sql.*;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Tip: when importing (mysqlimport) jwpl .txt files into db, use param --local or -L to make mysqlimport find the files!
 * Command: java -cp "farm/lib/data-assembly-0.1-SNAPSHOT.jar:farm/bin/spark-2.4.3-bin-hadoop2.7/jars/*" ix.data.wikipedia.BuildTables ensiwiki2020.hsf
 */
public class BuildTables {
    Wikipedia enwiki, siwiki;
    PagesTable pages;
    PairsTable pairs;
    private MediaWikiParser parser;
    static final Log log = LogFactory.getLog(BuildTables.class);
    SequenceFile.Writer writer;
    private long maxCount = -1;
    private boolean noStub = false;

    void init(String filename) throws ServiceException, WikiInitializationException, NetException, SQLException, IOException {
        //Sink database
        MySQLDatabase database = MySQLDatabase.apply("kommlabir01fl", "ensiwiki_2020").get();
        //MySQLDatabase database = MySQLHelper.loadDatabase("utwente", databaseName);
        pages = new PagesTable(database.connection());
        pages.create();
        pairs = new PairsTable(database.connection());
        pairs.create();

        //English wiki
        DatabaseConfiguration enConfig = new DatabaseConfiguration();
        enConfig.setDatabase("kommlabir01fl");
        enConfig.setHost("kommlabir01fl");
        enConfig.setDatabase("wiki_english_20200401");
        enConfig.setUser("wundt");
        enConfig.setPassword("PaSSWoRD");
        enConfig.setLanguage(WikiConstants.Language.english);
        enwiki = new Wikipedia(enConfig);

        //Simple wiki
        DatabaseConfiguration siConfig = new DatabaseConfiguration();
        siConfig.setDatabase("kommlabir01fl");
        siConfig.setHost("kommlabir01fl");
        siConfig.setDatabase("wiki_simple_20200401");
        siConfig.setUser("wundt");
        siConfig.setPassword("PaSSWoRD");
        siConfig.setLanguage(WikiConstants.Language.simple_english);
        siwiki = new Wikipedia(siConfig);

        //Parser
        MediaWikiParserFactory pf = new MediaWikiParserFactory();
        parser = pf.createParser();

        //SequenceFile writer
        open(filename);
    }

    void open(String filename) throws IOException {
        //SequenceFile writer
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        writer = SequenceFile.createWriter(fs, conf, new Path(filename), LongWritable.class, PageRecord.class);
    }

    public void close() throws IOException {
        writer.close();
    }

    private void create() throws SQLException {
        pages.create();
    }

    private void transfer() {
        Iterator<Page> siarticles = siwiki.getArticles().iterator();
        Page page;
        ParsedPage pp;
        String title = "";
        int counter = 0;
        while (siarticles.hasNext())
            try {

                //Page
                page = siarticles.next();
                title = page.getTitle().toString();

                // get a ParsedPage object
                pp = parser.parse(page.getText());

                // check for stub templates
                String stub = isStub(pp);
                if (!isNoStub() || stub == null) {
                    counter++;

                    // only the links to other Wikipedia language editions
                    long englishId = -1;
                    for (Link language : pp.getLanguages())
                        if (language.getTarget().startsWith("en:")) {
                            //System.out.println(language.getTarget());
                            try {
                                englishId = addEnglishPage(language.getTarget().substring(3)); }
                            catch (de.tudarmstadt.ukp.wikipedia.api.exception.WikiPageNotFoundException e) {
                                //Do nothing
                            }

                        }

                    if (englishId >= 0) {
                        long simpleId = pages.addPage("simple", page.getPageId(), stub, page.isDisambiguation(), page.isRedirect(), page.getText());
                        pairs.addPair(simpleId, englishId);
                        addRecord(simpleId, "simple", page.getPageId(), stub, page.isDisambiguation(), page.isRedirect(), page.getText());
                    }
                }
            } catch (Exception e) {
                log.warn("Error while adding page " + title, e);
            } finally {
                if (maxCount > 0 && counter > maxCount)
                    break;
            }
    }

    void addRecord(long id, String lang, int pageId, String stub, boolean disambiguation, boolean redirect, String text) throws IOException {
        PageRecord record = new PageRecord();
        record.put("id", id);
        record.put("lang", lang);
        record.put("wiki_id", pageId);
        record.put("stub", stub);
        record.put("disambiguation", disambiguation);
        record.put("redirect", redirect);
        record.put("wiki_text", text);
        writer.append(new LongWritable(id), record);
    }

    private String isStub(ParsedPage pp) {
        for (Template template : pp.getTemplates())
            if (template.getName().contains("stub"))
                return template.getName();
        return null;
    }

    private long addEnglishPage(String title) throws WikiApiException, SQLException, IOException {
        Page page = enwiki.getPage(title);
        ParsedPage pp = parser.parse(page.getText());
        String stub = isStub(pp);
        long englishId = pages.addPage("english", page.getPageId(), stub, page.isDisambiguation(), page.isRedirect(), page.getText());
        addRecord(englishId, "english", page.getPageId(), stub, page.isDisambiguation(), page.isRedirect(), page.getText());
        return englishId;
    }

    public boolean isNoStub() {
        return noStub;
    }

    public void setNoStub(boolean noStub) {
        this.noStub = noStub;
    }

    /**
     * First arg = filename, second arg = database name
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        BuildTables tables = new BuildTables();
        if (args.length > 2)
            if ("nostub".equalsIgnoreCase(args[1]))
                tables.setNoStub(true);
        if (args.length > 3)
            tables.maxCount = Long.parseLong(args[2]);
        tables.init(args[0]);
        tables.create();
        tables.transfer();
        tables.close();
    }

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
        private Connection connection;

        public PagesTable(Connection database) throws SQLException {
            this.connection = database;
            insert = database.prepareStatement("INSERT INTO pages (lang,wiki_id,stub,disambiguation,redirect,wiki_text) VALUES (?,?,?,?,?,COMPRESS(?))", PreparedStatement.RETURN_GENERATED_KEYS);
        }

        public boolean create() throws SQLException {
            String query = "CREATE TABLE IF NOT EXISTS pages (" +
                    "id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, " +
                    "lang ENUM(  'english',  'simple' ) NOT NULL, " +
                    "wiki_id BIGINT UNSIGNED NOT NULL, " +
                    "stub VARCHAR(255) NULL, " +
                    "disambiguation tinyint(2) default 0, " +
                    "redirect tinyint(2) default 0, " +
                    "wiki_text LONGTEXT NOT NULL" +
                    ") ENGINE = MYISAM;";
            Statement statement = connection.createStatement();
            return statement.execute(query);
        }

        public long addPage(String language, int wikiId, String stub, boolean disambiguation, boolean redirect, String text) throws SQLException {
            insert.setString(1, language);
            insert.setLong(2, wikiId);
            insert.setString(3, stub);
            insert.setBoolean(4, disambiguation);
            insert.setBoolean(5, redirect);
            //insert.setAsciiStream(6, org.apache.commons.io.IOUtils.toInputStream(text));
            insert.setString(6, text);
            insert.execute();
            ResultSet result = insert.getGeneratedKeys();
            if (result.next())
                return result.getLong(1);
            return -1;
        }

        public Connection getConnection() { return this.connection; }
    }

    private static class PairsTable {
        private PreparedStatement insert;
        private Connection connection;

        public PairsTable(Connection database) throws SQLException {
            this.connection = database;
            String q = "INSERT INTO pairs (simple_id, english_id) VALUES (?,?)";
            insert = database.prepareStatement(q);
        }

        public boolean create() throws SQLException {
            String query = "CREATE TABLE IF NOT EXISTS pairs (" +
                    "simple_id bigint(20) unsigned NOT NULL," +
                    "english_id bigint(20) unsigned NOT NULL" +
                    ") ENGINE=MyISAM;";
            Statement statement = connection.createStatement();
            return statement.execute(query);
        }

        public boolean addPair(long simpleId, long englishId) throws SQLException {
            insert.setLong(1, simpleId);
            insert.setLong(2, englishId);
            return insert.execute();
        }

        public Connection getConnection() { return this.connection; }
    }


}
