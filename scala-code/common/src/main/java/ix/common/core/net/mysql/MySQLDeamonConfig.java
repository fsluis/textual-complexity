package ix.common.core.net.mysql;

import ix.common.util.ConfigurationNode;
import ix.common.core.net.DeamonException;
import ix.common.core.net.Server;

/**
 * Data implementation using XML.
 * @version 1.0
 */
public class MySQLDeamonConfig implements MySQLDeamonData {
	private Server local;

	public void load(MySQLDeamon deamon, ConfigurationNode config) throws DeamonException {
        parseMySQLDeamon(deamon, config);
	}

	private void parseMySQLDeamon(MySQLDeamon deamon, ConfigurationNode tag) throws DeamonException {
		//port
        try{
            if (tag.hasValue("port"))
			    deamon.setPort(Integer.parseInt(tag.getValue("port")));
        } catch(NumberFormatException e) {
            throw new DeamonException("please learn how to type a number", e);
        }

        //load-file
        if(tag.hasChild("allow-load-file"))
            deamon.setAllowLoadFile(true);

        //properties
        if (tag.hasChild("properties"))
            deamon.setProperties(tag.getChild("properties").toProperties());

		//databases
        /*for (ConfigurationNode databaseNode : tag.getGrandChildren("database")) {
			MySQLDatabase mySqlDatabase = new MySQLDatabase(deamon);
			parseMySQLDatabase(mySqlDatabase, databaseNode);
			deamon.addDatabase(mySqlDatabase);
		}*/
	}

	/*private void parseMySQLDatabase(MySQLDatabase mySqlDatabase, ConfigurationNode config) throws DeamonException {
		if (!tag.hasAttributeValue("name"))
			throw new DeamonException("missing name attribute in mysqldatabasetag");
		mySqlDatabase.setName(tag.getAttributeValue("name"));
		mySqlDatabase.setProperties(parseProperties(tag));
	}*/

	/*private Properties parseProperties(Configuration tag) throws DeamonException {
		Properties properties = new Properties();
		while (tag.hasNextTag("property")) {
			XmlTag property = tag.getNextTag("property");
			if (!property.hasAttributeValue("name"))
				throw new DeamonException("missing name attribute in mysqlpropertytag");
			properties.put(property.getAttributeValue("name"), property.getValue());
		}
		while (tag.hasNextTag("remote")) {
			XmlTag remote = tag.getNextTag("remote");
			if (!remote.hasAttributeValue("server"))
				throw new DeamonException("missing server attribute in mysqlremotetag");
			if (remote.getAttributeValue("server").equals(local.getName()))
				properties.putAll(parseProperties(remote));
		}
		return properties;
	}*/
}
