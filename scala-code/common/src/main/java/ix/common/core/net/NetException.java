package ix.common.core.net;

/**
 * Thrown when an exception occurs in the .net package
 * @version 1.0
 */
public class NetException extends Exception {
	public NetException() {
	}

	public NetException(String msg) {
		super(msg);
	}

	public NetException(String msg, Throwable cause) {
		super(msg, cause);
	}

	public NetException(Throwable cause) {
		super(cause);
	}
}
