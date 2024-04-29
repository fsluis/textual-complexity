package ix.complexity.lucene3.esa;

/**
 * Whenever the services don't work.
 *
 * @author f
 * @version 1.0
 */
public class EsaException extends Exception {
    public EsaException() {
    }

    public EsaException(String msg) {
        super(msg);
    }

    public EsaException(String msg, Throwable cause) {
        super(msg, cause);
    }

    public EsaException(Throwable cause) {
        super(cause);
    }
}

