package ix.common.core.net;

/**
 * Thrown when an exception occurs in a deamon of a server
 * @version 1.0
 */
public class DeamonException extends Exception {
	public DeamonException() {
	}

	public DeamonException(String msg) {
		super(msg);
	}

	public DeamonException(String msg, Throwable cause) {
		super(msg, cause);
	}

	public DeamonException(Throwable cause) {
		super(cause);
	}
}
