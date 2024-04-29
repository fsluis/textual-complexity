package ix.common.core.services;

/**
 * Whenever the services don't work.
 *
 * @author f
 * @version 1.0
 */
public class ServiceException extends Exception {
	public ServiceException() {
	}

	public ServiceException(String msg) {
		super(msg);
	}

	public ServiceException(String msg, Throwable cause) {
		super(msg, cause);
	}

	public ServiceException(Throwable cause) {
		super(cause);
	}
}

