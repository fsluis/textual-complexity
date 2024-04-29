package ix.common.core.net.mysql;

import ix.common.util.ConfigurationNode;
import ix.common.core.net.DeamonException;

/**
 * Interface to handle the data for the mysql deamon.
 * 
 * @version 1.0
 */
public interface MySQLDeamonData {
    /**
     * This method should load the given MySQLDeamon with the configuration supplied. Of the deamon only the parent
     * Server instance has been set.
     * @param deamon empty deamon instance
     * @param config configure params supplied to the deamon
     * @throws DeamonException might something bad happen here...
     */
	public void load(MySQLDeamon deamon, ConfigurationNode config) throws DeamonException;
}
