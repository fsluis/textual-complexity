package ix.common.core.services;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.net.MalformedURLException;

/**
 * Services is a management system to keep a tree of services within your application.
 * The services have to be initialized by an data instance ({@link ServicesData}), which can be set at the root branch
 * using {@link ServiceBranch#setData(ServicesData)}.
 *
 *@version 1.4
 *@author f
 *@author m
 */
public class Services {
	private static Services instance; // keep the instance of this class, so that is will be persistent
	private ServiceBranch root; // the Root of the tree
    private boolean loaded = false;

	/**
	 * The constructor is private to prevent users to create an own instance of the Services tree.
	 * This forces them to use getInstance instead.
	 */
	private Services() {
		root = new ServiceBranch("root");
	}

	/**
	 * This will give an instance of the Services tree. This method is static because you are not allowed to
	 * construct a Services tree by yourself. Only one tree instance may be active in a running program
	 * @return a Services instance.
	 */
	private static Services getInstance() {
		if (instance == null)
			instance = new Services();
		return instance;
	}

    /**
     * Gets the root branch, from which there is one in the total JVM.
     * @return ServiceBranch instance, named "root"
     */
    public static ServiceBranch getRoot() {
        return getInstance().root;
    }

    /**
     * Loads the default root, with the default data handler (XMLServicesData) pointed to the default xml-file (resources/services.xml)
     * @return ServiceBranch instance, named "root"
     */
    public static ServiceBranch getDefault() throws ServiceException {
        ServiceBranch root = getRoot();
        if(getInstance().isLoaded())
            return root;
        XMLServicesData data = new XMLServicesData();
        try {
            data.setStream(findFile("services.xml"));
        } catch (Exception e) {
            throw new ServiceException("Couldn't find services settings file services.xml", e);
        }
        root.setData(data);
        getInstance().loaded = true;
        return root;
    }

    /**
     * Short-hand version using the default branch to retrieve a service of class klass.
     * @param klass
     * @return
     * @throws ServiceException
     */
    public static Service get(Class<? extends Service> klass) throws ServiceException {
        return getDefault().getService(klass);
    }

    /**
     * Searches popular locations for services xml
     * @return
     */
    protected static InputStream findFile(String uri) throws FileNotFoundException, MalformedURLException {
        // Local file, complete uri
        File file = new File(uri);
        if(file.exists())
            return new FileInputStream(file);
        // Local file in resources dir
        file = new File("resources/"+uri);
        if(file.exists())
            return new FileInputStream(file);
        // Resource in jar
        // Method one
        InputStream resource = ClassLoader.getSystemResourceAsStream(uri);
        if(resource!=null)
            return resource;
        // Method two
        resource = Services.class.getResourceAsStream(uri);
        if(resource!=null)
            System.out.println("three");
        // Method three
        resource = Thread.currentThread().getContextClassLoader().getResourceAsStream(uri);
        if(resource!=null)
            return resource;
        throw new FileNotFoundException("File not found: "+uri);
    }

    public boolean isLoaded() {
        return loaded;
    }
}