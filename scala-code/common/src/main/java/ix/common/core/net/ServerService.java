package ix.common.core.net;

import ix.common.util.ConfigurationNode;
import ix.common.core.services.Service;
import ix.common.core.services.ServiceException;

import java.util.HashMap;

/**
 * The serverserver-service is the server of the .net classes.
 * This class loads the servers and delivers the servers when
 * requested by other classes.
 * @version 1.0
 */
public class ServerService implements Service {
	private NetData parser;
	private HashMap<String, Server> servers = new HashMap<String, Server>();
	private Server local;

	//constructors
	/**
	 * Constructor.
	 */
	public ServerService() {
	}

	//get-methods
	/**
	 * Returns the server by the given name.
	 * @param name the name of the requested server.
	 * @throws NetException if the server isn't available.
	 * @return the requested server.
	 */
	public Server getServer(String name) throws NetException {
        if(local.getName().equals(name))
            return local;
		if (!servers.containsKey(name))
			servers.put(name, parser.getServer(name));
		return (Server) servers.get(name);
	}

	/**
	 * Returns the local server, the server on which this
	 * Java Virtual Machine is running.
	 * @return the local server.
	 */
	public Server getLocalServer() {
		return local;
	}

	//methods
	public void startService(ConfigurationNode config) throws ServiceException {
		//load parser
		if (!config.hasValue("data"))
			throw new ServiceException("ServerService service initparam data hasn't been set, ServerService service can't load");
		try {
			parser = (NetData) Class.forName(config.getValue("data")).newInstance();
		} catch(Exception e) {
			throw new ServiceException("unable to load ServerService service: can't load parser " + config.getValue("data"), e);
		}

		try {
			parser.configure(config);

			//load local server
			local = parser.getLocalServer();
			servers.put(local.getName(), local);
		} catch(NetException e) {
			throw new ServiceException(e);
		}
	}

	public void stopService() throws ServiceException {
	}
}
