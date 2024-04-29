package ix.common.core.net;

import ix.common.util.ConfigurationNode;

/**
 * The deamon interface represents a deamon running on a server.
 * This interface should be implemented when you want your deamon
 * to be accessible through a Server-instance. You should also
 * specify the implementing class to the parser so it knows to load
 * the deamon for the server on which it should run.
 * @version 1.0
 */
public interface Deamon {
	/**
	 * Configs the deamon.
	 * @param config the configuration settings as loaded by the parser for this deamon
	 * @throws DeamonException when any configuration/loading error occurs.
	 */
	public void configDeamon(Server parent, ConfigurationNode config) throws DeamonException;

	/**
	 * Returns the for the server unique name by which the deamon is identified on the server.
	 * @return the for the server unique name of this deamon.
	 */
	public String getName();
}
