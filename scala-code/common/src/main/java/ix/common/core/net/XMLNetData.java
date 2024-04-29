package ix.common.core.net;


import ix.common.util.ConfigurationNode;
import ix.common.core.xml.Document;
import ix.common.core.xml.XmlTag;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Hashtable;
import java.util.List;

/**
 * This class provides a way of retrieving settings for the .net package
 * from .xml files, using the {@link ix.common.core.xml} package.
 * <p>
 * The configuration of this data class takes three parse parameters:
 * <ul>
 *	<li>local-server-xml
 *	<li>drivers-xml
 *	<li>servers-xml
 * </ul>
 * </p><p>
 *
 * <b>local-server-xml</b><br />
 * This is the full filename (including the full path) of the local server.
 * This filename is for every Servlet different, so this should be given
 * by the Servlet. For the syntax of a server-xml you should look at
 * the server-xml topic.<br />
 * </p><p>
 *
 * <b>drivers-xml</b><br />
 * This is the project-relative filename (so the part of the filename without
 * the path to the project dir) to the drivers-xml. This file contains
 * a list of classes that implement the ...
 * interface. The syntax of this file is as follows, example:<br />
 * </p><p>
 * &lt;drivers&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;driver&gt;nl.isolation.util.sql.mysql.MySqlDriver&lt;/driver&gt;<br />
 * &lt;/drivers&gt;<br />
 * </p><p>
 * where &lt;driver&gt;&lt;/driver&gt; can be repeated multiple (0 to *) times.
 * </p><p>
 *
 * <b>servers-xml</b><br />
 * This is the project-relative filename to an xml-file containing a list
 * of all servers that can be used by this project. The syntax of this file
 * is as follows, example:<br />
 * </p><p>
 * &lt;servers&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;server name="Spider"&gt;resources/configDeamon/net/spider/server.xml&lt;/server&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;server name="Kennel"&gt;resources/configDeamon/net/kennel/server.xml&lt;/server&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;server name="Laptop"&gt;resources/configDeamon/net/laptop/server.xml&lt;/server&gt;<br />
 * &lt;/servers&gt;<br />
 * </p><p>
 * Where the &lt;server&gt;&lt;/server&gt; tags can be repeated multiple times. The
 * name attribute of this tag gives an unique identifier to the server,
 * where the value of this tag is the filename of the server-xml.
 * </p><p>
 *
 * <b>server-xml</b><br />
 * This is the project-relative filename pointing to an xml-file containing
 * specific information about a server. The syntax is as follows, example:
 * </p><p>
 * &lt;server&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;name&gt;Kennel&lt;/name&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;host&gt;supermarkt.student.utwente.nl&lt;/host&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;project-dir&gt;D:/Kennel/tfd-xp/&lt;/project-dir&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;deamon&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;deamon-class&gt;ix.common.core.net.HttpDeamon&lt;/deamon-class&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;init-param&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;param-name&gt;something&lt;/param-name&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;param-value&gt;tfd-xp/&lt;/param-value&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;/init-param&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;init-param&gt; <br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;param-name&gt;somethingelse&lt;/param-name&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;param-value&gt;D:/Kennel/tfd-xp/&lt;/param-value&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;/init-param&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;init-param&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;param-name&gt;port&lt;/param-name&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;param-value&gt;8080&lt;/param-value&gt;<br />
 * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;/init-param&gt;<br />
 * &nbsp;&nbsp;&nbsp;&lt;/deamon&gt;<br />
 * &lt;/server&gt;<br />
 * </p><p>
 * Where
 * <ul>
 * 	<li>name is the unique identifier of the server, which should equal the
 * name specified in the servers-xml. See {@link Server#getName()},
 * 	<li>host is the default host of this server. See {@link Server#getHost()},
 * 	<li>and project-dir points to the root directory of this project. See {@link Server#getProjectDir()}.
 * </ul>
 * The deamon tag can be repeated multiple time to add deamons to this server,
 * and exists of the following elements:
 * <ul>
 * 	<li>Deamon-class, containing the classpath of the class of this deamon.
 * 	The given class should implement the Deamon class.
 * 	<li>Init-param, can be used multiple times with unique param-names.
 * 	Contains the tags param-name, which specifies the unique name of the
 *	param, and param-value, which specifies the value of this param.
 *	The params are given to the {@link Deamon#configDeamon(Server, ConfigurationNode)}
 *	method of the class implementing the Deamon interface.
 * </ul>
 * </p>
 *
 *
 * @version 1.0
 */
public class XMLNetData implements NetData {
	private Hashtable<String, String> servers = new Hashtable<String, String>();
	private Server local = new Server();
    private String localXml;

    public XMLNetData() {
	}

	public void configure(ConfigurationNode sc) throws NetException {
		local.setLocal(true);
        loadServers(sc);
		parseServer(local, localXml);
	}

	public Server getServer(String name) throws NetException {
        name = name.toLowerCase();
		if (!servers.containsKey(name))
			throw new NetException("can't load unknown server " + name + ", please check if this server has been set in the servers-xml (working dir: "+System.getProperty("user.dir")+")");
		Server server = new Server();
		parseServer(server, servers.get(name));
		return server;
	}

	public Server getLocalServer() throws NetException {
		return local;
	}

	private void loadServers(ConfigurationNode config) throws NetException {
        String localhost = null;
        try {
            localhost = InetAddress.getLocalHost().getHostName();
        } catch (UnknownHostException e) {
            throw new NetException(e);
        }

        List<ConfigurationNode> nodes = config.getGrandChildren("server");
		for (ConfigurationNode node : nodes) {
            String hostname = node.getValue("hostname");
            String xmlFile = node.getValue("xml-file");
            String name = node.getValue("name");
			if ( (name==null) || (xmlFile==null) || hostname==null )
				throw new NetException("hostname or or name or xml-file attribute not set in server tag");
			servers.put(name.toLowerCase(), xmlFile);
            if(hostname.equals(localhost))
                localXml = xmlFile;
		}

        if(localXml == null)
            throw new NetException("Can't find localhost "+localhost+" in servers xml");
	}

	private void parseServer(Server server, String xmlfile) throws NetException {
		ConfigurationNode config;
		try {
			XmlTag tag = new Document(xmlfile);
            config = ConfigurationNode.parseFromXml(tag);
		} catch(Exception e) {
			throw new NetException("couldn't load xmlfile " + xmlfile, e);
		}

		if ((!config.hasValue("host")) ||
			(!config.hasValue("name"))
        )
			throw new NetException("Either <name> or <host> tag not set, server cannot load");

		server.setName(config.getValue("name"));
		server.setHost(config.getValue("host"));

		for (ConfigurationNode deamonNode : config.getGrandChildren("deamon"))
			server.addDeamon(parseDeamon(server, deamonNode));
	}

	private Deamon parseDeamon(Server server, ConfigurationNode config) throws NetException {
		if (!config.hasValue("class"))
			throw new NetException("<class> tag not set in deamon-tag, server " + server.getName() + " cannot load");

		try {
			Deamon deamon = (Deamon) Class.forName(config.getValue("class")).newInstance();
			deamon.configDeamon(server, config.getChild("configuration"));
			return deamon;
		} catch(Exception e) {
			throw new NetException("couldn't load deamon", e);
		}
	}
}
