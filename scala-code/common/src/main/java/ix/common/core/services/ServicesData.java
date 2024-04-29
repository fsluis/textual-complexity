package ix.common.core.services;

/**
 * Implement this to handle the loading/saving of the services.
 * 
 * @author f
 * @version 1.0
 */
public interface ServicesData {
    /**
     * Load the data to the given branch. The supplied branch will always be the root branch, for which all
     * subbranches and leaves should be loaded.
     * @param branch the branch to load
     * @throws ServiceException any error
     */
    void load(ServiceBranch branch) throws ServiceException;

    /**
     * Save the given leaf.
     * @param leaf leaf
     * @return true if indeed the data has changed, false if it hasn't (which could mean that the
     *  saved object is already the same in the data)
     * @throws ServiceException
     */
    boolean save(ServiceLeaf leaf) throws ServiceException;

    /**
     * Save the given branch.
     * @param branch
     * @return true if indeed the data has changed, false if it hasn't (which could mean that the
     *  saved object is already the same in the data)
     * @throws ServiceException
     */
    boolean save(ServiceBranch branch) throws ServiceException;

    boolean install() throws ServiceException;
}
