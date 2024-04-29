package ix.common.core.services;

import ix.common.util.ConfigurationNode;

/**
 * A service is a class that can be started and passed a configuration, and it can be stopped.
 * Maybe some status support should be added in the future.
 *@version 1.2
 *@author f
 *@author m
 */
public interface Service {
	public final static int LOADED = 0;
	public final static int STARTED = 1;
	public final static int STOPPED = 2;

	/**
	 * Start the service. The class should be initializing its variables at this point, an read its information.
	 * @throws ServiceException this exception can be thrown if the service is not able to start (like wrong configuration)
	 */
	public void startService(ConfigurationNode config) throws ServiceException;

	/**
	 * Stop the service. The class should be destroying its variables, and store all it's active information.
	 * @throws ServiceException this exception can be thrown if the service is not able to stop or cannot be stopped.
	 */
	public void stopService() throws ServiceException;
}