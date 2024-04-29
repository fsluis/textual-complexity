package ix.common.core.services;

import ix.common.util.ConfigurationNode;

/**
 * Leaf wrapping a Service, controlling the lifecycly of the service.
 *
 * @version 1.1
 * @author m
 * @author f (MaX version)
 */
public class ServiceLeaf {
	private Service service;
	private ConfigurationNode configuration;
	private boolean running = false;
    private ServiceBranch branch;

	/**
     * Create new leaf with a service and configure.
     * @param service the service
     * @param configuration the configure
     */
    public ServiceLeaf(Service service, ConfigurationNode configuration) {
        setService(service);
        setConfiguration(configuration);
	}

    /**
     * Create new leaf from Classname (should be Service-instance) and configure.
     * @param className Service class name
     * @param configuration configure
     * @throws ServiceException if the class can't be retrieved or isn't a Service instance
     */
    public ServiceLeaf(String className, ConfigurationNode configuration) throws ServiceException {
        setService(className);
        setConfiguration(configuration);
    }

    /**
     * Create new empty leaf.
     */
    protected ServiceLeaf() {}

	private void setService(Service service) {
		this.service = service;
		running = false;
	}

	private void setService(String className) throws ServiceException {
		try {
			setService((Service) Class.forName(className).newInstance());
		} catch(Exception e) {
			throw new ServiceException("unable to load class: " + className, e);
		}
	}

	/**
     * Returns the class of this Service.
     * @return class of the Service
     */
    public Class getServiceClass() {
		return service.getClass();
	}

	private void setConfiguration(ConfigurationNode configuration) {
		this.configuration = configuration;
	}

	/**
     * The configure given to startService on the Service instance.
     * @return configure
     */
    public ConfigurationNode getConfiguration() {
		return configuration;
	}

    /**
     * The branch to which this service belongs. This value is being set when the leaf is added to the branch.
     * @return the branch or null if it hasn't been added to a branch
     */
    public ServiceBranch getBranch() {
        return branch;
    }

    void setBranch(ServiceBranch branch) {
        this.branch = branch;
    }

	/**
     * Check if the Service is running.
     * @return true if it is, false if it isn't
     */
    public boolean isRunning() {
		return running;
	}

	/**
     * Get the Service instance.
     * @return Service instance
     * @throws ServiceException if the service can't be started
     */
    public Service getService() throws ServiceException {
		if (!isRunning()) startService();
		return service;
	}

	/**
     * Starts the service.
     * @throws ServiceException when it doesn't work!
     */
    private synchronized void startService() throws ServiceException {
		if (isRunning()) return;
		service.startService(configuration);
		running = true;
	}

    public void save() throws ServiceException {
        if(branch==null)
            throw new ServiceException("Please add this leaf to a branch before saving");
        branch.getData().save(this);
    }

	public String toString() {
		return getServiceClass().toString();
	}
}