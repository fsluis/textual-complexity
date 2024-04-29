package ix.common.data.wikipedia;

import de.tudarmstadt.ukp.wikipedia.api.exception.WikiInitializationException;
import ix.common.core.net.NetException;
import ix.common.core.services.ServiceException;

import java.io.IOException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 2/18/12
 * Time: 10:44 PM
 * To change this template use File | Settings | File Templates.
 */
public class BuildTables20k extends BuildTables {
    String filename;

    @Override
    void init(String filename) throws ServiceException, WikiInitializationException, NetException, SQLException, IOException {
        this.filename = filename;
        filename+=".1.hsf";
        super.init(filename);
    }

    public void transfer() throws SQLException, IOException {
        PreparedStatement englishStatement = this.pages.getConnection().prepareStatement("SELECT id, lang, wiki_id, stub, disambiguation, redirect, UNCOMPRESS(wiki_text) AS wiki_text FROM view_pages_english_age_20000");
        PreparedStatement simpleStatement = this.pages.getConnection().prepareStatement("SELECT id, lang, wiki_id, stub, disambiguation, redirect, UNCOMPRESS(wiki_text) AS wiki_text FROM view_pages_simple_age_20000");
        englishStatement.setFetchSize(100);
        simpleStatement.setFetchSize(100);
        ResultSet english = englishStatement.executeQuery();
        ResultSet simple = simpleStatement.executeQuery();
        int go = 2, n = 0;
        while(go>0) {
            go=0;

            if(english.next()) {
                addRecord(english);
                go++;
            }
            
            if(simple.next()) {
                addRecord(simple);
                go++;
            }

            if(++n%100==0)
                writer.sync();

            if(++n%10000==0) {
                close();
                open(filename+"."+(n/10000+1)+".hsf");
            }
        }
    }

    //void addRecord(long id, String lang, int pageId, String stub, boolean disambiguation, boolean redirect, String text) throws IOException {
    /*
                    "id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, " +
                    "lang ENUM(  'english',  'simple' ) NOT NULL, " +
                    "wiki_id BIGINT UNSIGNED NOT NULL, " +
                    "stub VARCHAR(255) NULL, " +
                    "disambiguation tinyint(2) default 0, " +
                    "redirect tinyint(2) default 0, " +
                    "wiki_text LONGTEXT NOT NULL" +

     */
    private void addRecord(ResultSet r) throws SQLException, IOException {
        addRecord(
                r.getLong("id"),
                r.getString("lang"),
                r.getInt("wiki_id"),
                r.getString("stub"),
                r.getBoolean("disambiguation"),
                r.getBoolean("redirect"),
                r.getString("wiki_text")
                );
    }

    public static void main(String[] args) throws IOException, WikiInitializationException, SQLException, NetException, ServiceException {
        BuildTables20k build = new BuildTables20k();
        build.init("resources/pages_nostub_20k");
        build.transfer();
        build.close();
    }
}
