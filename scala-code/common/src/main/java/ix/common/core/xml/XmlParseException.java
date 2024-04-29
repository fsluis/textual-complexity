package ix.common.core.xml;

public class XmlParseException extends Exception {

	public XmlParseException() {
	}

	public XmlParseException(String msg) {
		super(msg);
	}

	public XmlParseException(String msg, Throwable cause) {
		super(msg, cause);
	}

	public XmlParseException(Throwable cause) {
		super(cause);
	}
}
