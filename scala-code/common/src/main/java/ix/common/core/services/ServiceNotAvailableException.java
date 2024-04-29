package ix.common.core.services;

/**
 * Whenever the Service isn't available
 */
public class ServiceNotAvailableException extends RuntimeException {
	public ServiceNotAvailableException() {
	}

	public ServiceNotAvailableException(String msg) {
		super(msg);
	}

	public ServiceNotAvailableException(String msg, Throwable cause) {
		super(msg, cause);
	}

	public ServiceNotAvailableException(Throwable cause) {
		super(cause);
	}
}

