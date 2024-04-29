package ix.common.core.services;


import ix.common.util.ConfigurationNode;

import java.util.Enumeration;
import java.util.Hashtable;


/**
 * A ServiceBranch is a branch in the Services tree. This class can contain other branches, and Services (leaves).
 * @version 1.1
 * @author m
 * @author f (MaX version)
 */
public class ServiceBranch {

	private Hashtable<Class, ServiceLeaf> leaves = new Hashtable<Class, ServiceLeaf>(); // the table with the available services
	private Hashtable<String, ServiceBranch> branches = new Hashtable<String, ServiceBranch>(); // service subgroups
	private String name = "";
    private ServiceBranch parent;
    private ServicesData data;
    private int id;

	/**
	 * default constructor. The name of the ServiceBranch should always be set, so don't forget to call the setName() function.
	 */
	public ServiceBranch(String name) {
		this.name = name;
	}

    /**
     * default constructor.
     */
    public ServiceBranch() {
    }

    /**
     * Optional id.
     * @return id
     */
    public int getId() {
        return id;
    }

    /**
     * Optional id.
     * @param id
     */
    public void setId(int id) {
        this.id = id;
    }

    /**
     * Set the name.
     * @param name
     */
    public void setName(String name) {
        this.name = name;
    }

	/**
	 * gets the name of this ServiceBranch this is also the name used in paths in reference to services within this branch.
	 *@return the name of this branch, in lowercase.
	 */
	public String getName() {
		return name.toLowerCase();
	}

    /**
     * The parent branch of this branch, if any.
     * @return ServiceBranch instance or null
     */
    public ServiceBranch getParent() {
        return parent;
    }

    /**
     * Checks if this branch has a parent.
     * @return true or false
     */
    public boolean hasParent() {
        return (parent!=null);
    }

    private void setParent(ServiceBranch parent) {
        this.parent = parent;
    }

	/**
	 * returns an Enumeration of service classes that are contained in this branch
	 *@return an Enumeration of service classes that are contained in this branch
	 */
	private Enumeration getLeafClasses() {
		return leaves.keys();
	}

	/**
	 * checks wether a branch (directory) with the given name is in this branch
	 *@param branchName the name of the directory that must be contained in this branch
	 *@return true if the directory is in this branch, false otherwise
	 */
	public boolean containsBranch(String branchName) {
		return branches.containsKey(branchName.toLowerCase());
	}

	/**
	 * returns the branch with the given name
	 *@param branchName the name of the directory that must be returned
	 *@return ServiceBranch if the branch is in this branch, null otherwise
	 */
	public ServiceBranch getBranch(String branchName) {
		return (ServiceBranch) branches.get(branchName.toLowerCase());
	}

	/**
	 * Add a branch to this branch. If the branch is already within this branch, the two branches will be merged.
     * The added branch it's parent will be set to this branch.
	 *@param branch branch to add to this branch
	 *@throws ServiceException if there was a conflict in merging with an existing branch
	 */
	public void addBranch(ServiceBranch branch) throws ServiceException {
		if (containsBranch(branch.getName())) {
			ServiceBranch existingBranch = getBranch(branch.getName());
			existingBranch.merge(branch);
		} else {
			branches.put(branch.getName(), branch);
            branch.setParent(this);
		}
	}

	/**
	 * returns an Enumeration of directorynames that are contained in this branch
	 *@return an Enumeration of directorynames that are contained in this branch
	 */
	private Enumeration getBranchNames() {
		return branches.keys();
	}

	/**
	 * checks wether a leaf (Service) with the given name is in this branch
	 *@param leafClass the Service-Class of the leaf that must be contained in this branch
	 *@return true if the service is in this branch, false otherwise
	 */
	boolean containsLeaf(Class leafClass) {
		return leaves.containsKey(leafClass);
	}

	/**
     * Returns the leaf with class leafClass.
     * @param leafClass the Service class
     * @return the ServiceLeaf
     */
    public ServiceLeaf getLeaf(Class leafClass) {
		return (ServiceLeaf) leaves.get(leafClass);
	}

	/**
	 * Adds a leaf to this branch
	 *@throws ServiceException an ServiceException is thrown if a leaf with the same name exists in this branch
	 */
	public void addLeaf(ServiceLeaf leaf) throws ServiceException {
		if (containsLeaf(leaf.getServiceClass()))
			throw new ServiceException("there already is a service loaded of class " + leaf.getServiceClass() + "! Please make sure that the names of the services aren't equal.");
        leaf.setBranch(this);
		leaves.put(leaf.getServiceClass(), leaf);
	}

	/**
	 * Retrieves a Service from this branch.
	 *@throws ServiceNotAvailableException if the requested service does not exist in this branch
	 *@return the requested Service
	 */
	public Service getService(Class serviceClass) throws ServiceNotAvailableException {
		if (containsLeaf(serviceClass)) {
			try {
				ServiceLeaf leaf = (ServiceLeaf) leaves.get(serviceClass);
				return leaf.getService();
			} catch(ServiceException se) {
				throw new ServiceNotAvailableException(se);
			}
		}
		throw new ServiceNotAvailableException("the service " + serviceClass + " is not available");
	}

    /**
     * Checks if the service is available in this branch.
     * @param serviceClass the class of the service
     * @return true if exists, false otherwise
     */
    public boolean hasService(Class serviceClass) {
        return containsLeaf(serviceClass);
    }

	/**
	 * Retrieves a Service from this branch.
	 *@throws ServiceNotAvailableException if the requested service does not exist in this branch
	 *@return the requested Service
	 */
	public ConfigurationNode getConfiguration(Class serviceClass) throws ServiceNotAvailableException {
		if (containsLeaf(serviceClass)) {
			ServiceLeaf leaf = (ServiceLeaf) leaves.get(serviceClass);
			return leaf.getConfiguration();
		}
		throw new ServiceNotAvailableException("the service " + serviceClass + " is not available");
	}

	/**
	 * merges services and subbranches of the given branch with this branch.
	 *@param mergeWith the ServiceBranch that will be merged with this branch.
	 *@throws ServiceException if there were conflicts during the merge process.
	 */
	private void merge(ServiceBranch mergeWith) throws ServiceException {
		if (mergeWith == null) throw new IllegalArgumentException("mergeWith must not be null!");
		boolean conflict = false;

        /** merge the leaves */
		String conflictMessage = "";
        Enumeration leafClasses = mergeWith.getLeafClasses();
        while(leafClasses.hasMoreElements()) {
			Class leafClass = (Class)leafClasses.nextElement();
			try {
				addLeaf(mergeWith.getLeaf(leafClass));
			} catch(ServiceException se) {
				conflictMessage = se.getMessage();
				conflict = true;
			}
		}
		/** merge the branches */
        Enumeration branchNames = mergeWith.getBranchNames();
		while (branchNames.hasMoreElements()) {
			String branchName = (String) branchNames.nextElement();
			try {
				addBranch(mergeWith.getBranch(branchName));
			} catch(ServiceException se) {
				conflictMessage = se.getMessage();
				conflict = true;
			}
		}

		if (conflict)	throw new ServiceException("there were conflicts during the merge! (" + name + " & " +
			mergeWith.getName() + " ): " + conflictMessage);
	}

    /**
     * Sets the data instance for this branch and it's subsequent subbranches. Also loads the data immediately.
     * Only one instance can be setted for the whole tree of branches.
     * @param data ServicesData instance, may not be null!
     * @throws ServiceException if something goes wrong
     */
    public void setData(ServicesData data) throws ServiceException {
        this.data = data;
        data.load(this);
    }

    /**
     * Gets the data implementation used.
     * @return data implementation
     * @throws ServiceException if none is available
     */
    ServicesData getData() throws ServiceException {
        if(hasParent())
            return getParent().getData();
        else if(data!=null)
            return data;
        else
            throw new ServiceException("No ServiceData instance available on this ServiceBranch or it's parents");
    }

    /**
     * Saves this branch.
     * @throws ServiceException when saving didn't work
     */
    public void save() throws ServiceException {
        getData().save(this);
    }

	public String toString() {
		String result = getName() + " (";
        Enumeration leafClasses = getLeafClasses();
		while (leafClasses.hasMoreElements()) {
			Object leafClass = leafClasses.nextElement();
			result += leafClass.toString();
			if (leafClasses.hasMoreElements()) result += ", ";
		}
		result += ":";
		for (Enumeration leaveNames = getBranchNames(); leaveNames.hasMoreElements();) {
			String leafName = (String) leaveNames.nextElement();
			result += getBranch(leafName).toString();
			if (leaveNames.hasMoreElements()) result += ", ";
		}
		result += ")";
		return result;
	}
}