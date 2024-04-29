package ix.common.core.net;

import ix.common.util.ConfigurationNode;

/**
 * Represents the mail settings of a server.
 * <p>
 * This deamon requires the following parse parameters:
 * <ul>
 * 	<li>administrator-email: {@link MailDeamon#getAdministratorEmail()}
 *	<li>smtp-address: {@link MailDeamon#getSmtpAddress()}
 * </lu>
 * @author m
 * @version 1.0
 */
public class MailDeamon implements Deamon {
	public final static String NAME = "MAIL";
	private String smtpAddress;
	private String administratorEmailAddress;
    private Server parent;

	//constructors
	/**
	 * Constructor.
	 */
	public MailDeamon() {
	}

	public String getSmtpAddress() {
		return smtpAddress;
	}

	public String getAdministratorEmail() {
		return administratorEmailAddress;
	}

    public String getName() {
        return NAME;
    }

	//methods
	public void configDeamon(Server parent, ConfigurationNode config) throws DeamonException {
		if (!config.hasValue("administrator-email"))
			throw new DeamonException("administrator-email param not set, maildeamon can't start");
		if (!config.hasValue("smtp-address"))
			throw new DeamonException("smtp-address param not set, maildeamon can't start");

		this.smtpAddress = config.getValue("smtp-address");
		this.administratorEmailAddress = config.getValue("administrator-email");
        this.parent = parent;
	}
}
