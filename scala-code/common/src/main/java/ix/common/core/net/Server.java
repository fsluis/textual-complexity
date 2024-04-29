package ix.common.core.net;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.List;
import java.util.Vector;

/**
 * Represents a server.
 * @version 1.0
 */
public class Server {
	private String name, projectDir, tempDir, host;
	private Hashtable deamons = new Hashtable();
	private InetAddress address = null;
	private boolean local = false;

	//constructors
	/**
	 * Constructor.
	 */
	Server() {
	}

	//set-queries
	/**
	 * Sets the root dir of the project for which this MaX instance is running.
	 * @param projectDir the root dir of the project
	 */
	void setProjectDir(String projectDir) {
		if (!projectDir.endsWith("/"))
			projectDir = projectDir.concat("/");
		this.projectDir = projectDir;
	}

	/**
	 * Sets the unique name of this server.
	 * @param name the name of this server.
	 */
	void setName(String name) {
		this.name = name;
	}

	/**
	 * Sets the host of this server.
	 * @param host of this server.
	 */
	void setHost(String host) {
		this.host = host;
	}

	/**
	 * Adds a deamon to this server.
	 * @param deamon the deamon to be added,
	 */
	void addDeamon(Deamon deamon) {
		deamons.put(deamon.getName(), deamon);
	}

	/**
	 * Sets if this server is the local server on which this
	 * Java Virtual Machine is running.
	 * @param local true if this is the local server, false otherwise.
	 */
	void setLocal(boolean local) {
		this.local = local;
	}

    /**
     * Sets the temp dir of this server.
     * @param tempDir
     */
    void setTempDir(String tempDir) {
        this.tempDir = tempDir;
    }

	//get-queries
	/**
	 * Gets the name of this server.
	 * @return the name of this server.
	 */
	public String getName() {
		return name;
	}

	/**
	 * Gets the hostname of this server.
	 * @return the hostname of this server.
	 */
	public String getHost() {
		return host;
	}

	/**
	 * Gets the internet-address of this server.
	 * @return the InetAddress
	 */
	public InetAddress getAddress() throws UnknownHostException {
		if (this.address == null)
			address = InetAddress.getByName(host);
		return address;
	}

	/**
	 * Gets the root directory of this MaX project for this server. The projectDir is
	 * the home-directory + the user directory + the project directory.
	 * @return the projectDir.
	 */
	public String getProjectDir() {
		return projectDir;
	}

    /**
     * Gets the temporary directory on this server.
     * @return the temp dir
     */
    public String getTempDir() {
        return tempDir;
    }

	/**
	 * Gets if the server is the local server on which this Java
	 * Virtual Machine is running.
	 * @return true if this is the local server, false otherwise.
	 */
	public boolean isLocal() {
		return local;
	}

	/**
	 * Gets a deamon running on this server.
	 * @param name the name of the deamon to be retrieved.
	 * @throws NetException if the deamon isn't available on this server.
	 * @return the requested deamon
	 */
	public Deamon getDeamon(String name) throws NetException {
		if (!deamons.containsKey(name))
			throw new NetException("requested deamon " + name + " isn't available on server " + getName());
		return (Deamon) deamons.get(name);
	}

	/**
	 * Gets the deamons of type type running on this server.
	 * @param type the class of which the deamons must be.
	 * @return the requested deamons.
	 */
	public List getDeamons(Class type) {
		Enumeration deamons = this.deamons.elements();
		Vector result = new Vector();
		while (deamons.hasMoreElements()) {
			Deamon deamon = (Deamon) deamons.nextElement();
			if (type.isInstance(deamon))
				result.add(deamon);
		}
		return result;
	}

	//methods
	/**
	 * Compares this server with another server-instance.
	 * @return true if the names (getNickName() of both
	 * Server-instances are equal to each other, false otherwise.
	 */
	public boolean equals(Server server) {
		if (server == null) return false;
		if (getName().equals(server.getName())) return true;
		return false;
	}
}
