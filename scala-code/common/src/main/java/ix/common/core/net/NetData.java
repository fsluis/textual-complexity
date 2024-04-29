package ix.common.core.net;

import ix.common.util.ConfigurationNode;


/**
 * This interface specifies the methods that should
 * be supported by the .net parser.
 * The parser is responsible for loading the .net servers
 * and deamons.
 * @version 1.0
 */
public interface NetData {
	/**
	 * Configures this parser by the init-params given in the
	 * .net service configuration.
	 * @throws NetException if the configuration/loading of the
	 * parser fails.
	 */
	void configure(ConfigurationNode sc) throws NetException;

	/**
	 * Should return an iterator pointing to instances of the
	 * interface Driver. These drivers will be loaded.
	 * @throws NetException if anything goes wrong.
	 * @return iterator pointing to instances of the
	 * interface Driver
	 */
	//Iterator getDrivers() throws NetException;

	/**
	 * Should return an instance of a server.
	 * @param name is the name of the requested server
	 * @throws NetException if the server isn't available
	 * for loading or if anything else goes wrong.
	 * @return an instance of the requested server.
	 */
	Server getServer(String name) throws NetException;

	/**
	 * Should return the local server. That is, the server on
	 * which this Java Virtual Machine is running.
	 * @throws NetException if anything goes wrong.
	 * @return the local server.
	 */
	Server getLocalServer() throws NetException;
}
