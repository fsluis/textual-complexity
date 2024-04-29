package ix.common.core.xml;

/**
 * Definition of an Xml node
 * @author m
 * @version 1.2
 */
public interface XmlNode {

	public static final int TYPE_TAG = 1;
	public static final int TYPE_TEXT = 2;
	public static final int TYPE_COMMENT = 3;

  /**
	 * returns the type of node
	 * @return the type of node
	 * @see #TYPE_COMMENT
	 * @see #TYPE_TAG
	 * @see #TYPE_TEXT
	 */
	public int nodeType();

	/**
	 * returns the string content of the xml node
	 * @return the string content of the xml node
	 */
	public String getContent();

    public String getName();

}
