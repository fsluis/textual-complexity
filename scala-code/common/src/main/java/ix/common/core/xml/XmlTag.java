package ix.common.core.xml;

import java.util.Map;

/**
 * XMLTag is the interface representating a Tag from a parsed <code>Document</code>.
 * This class enables browsing through a parsed XML document, and is made as a wrapper for the Xerces project.
 * This class allows easier parsing and checking for values.
 *
 * @author m
 * @version 1.6
 */
public interface XmlTag extends XmlNode {

	/**
	 * Checks wether there is a next element available with the tagname <code>tagname</code>.
	 * @param tagname the name of the tag to be checked on availability.
	 * @return <code>true</code> if there is a next tag available with the tagname <code>tagname</code>;
	 *         <code>false</code> otherwise.
	 */
	public boolean hasNextTag(String tagname);

	/**
	 * Returns the next available <code>XMLTag</code> with the tagname <code>tagname</code>
	 * @param tagname the name of the tag to be returned.
	 * @return the next available <code>XMLTag</code> with the tagname <code>tagname</code>, <code>null</code> if the tag was not available.
	 * @see	#hasNextTag(String)
	 */
	public XmlTag getNextTag(String tagname);

	/**
	 * Returns the value of an attribute of a tag. Use the <code>defaultvalue</code> parameter to specify the returned value if the attribute was not found.
	 * @param attribute the name of the attribute to return the value of
	 * @param defaultvalue the value that will be returned if the attribute was not found
	 * @return the value of an attribute of a tag. Use the <code>defaultvalue</code> parameter to specify the returned value if the attribute was not found.
	 * @see	#getAttributeValue(String attribute)
	 */
	public String getAttributeValue(String attribute, String defaultvalue);

	/**
	 * Returns the value of an attribute of a tag. <code>attributename="attributevalue"</code>. To get this value, use getAttributeValue("attributename").
	 * @param attribute the name of the attribute to return the value of
	 * @return the value of an attribute of a tag. <code>attributename="attributevalue"</code>. To get this value, use getAttributeValue("attributename").
	 *         <code>null</code> is returned if the specified attribute was not found.
	 * @see	#getAttributeValue(String attribute, String defaultvalue)
	 */
	public String getAttributeValue(String attribute);

	/**
	 * Checks wether the attribute with the tagname <code>attribute</code> is available.
	 * @param attribute the name of the attribute to be checked on availability.
	 * @return <code>true</code> if the attribute is available
	 * @return <code>false</code> otherwise.
	 */
	public boolean hasAttributeValue(String attribute);

	/**
	 * Returns a Map of attributes of a tag.
	 * @return a Map of attributes of a tag.
	 * @since 1.2
	 */
	public Map getAttributes();

	/**
	 * Returns the textual value of the element. <code>&lt;mytag&gt;mytag-text value&lt;/mytag&gt;</code>.
	 * @return the textual value of the element; an empty <code>String</code> if the element did not contain any text
	 */
	public String getValue();

	/**
	 * Returns wether this XmlTag has an value or not.
	 * @return <code>true</code> if the Tag has an textvalue, <code>false</code> otherwise
	 * @since 1.3
	 */
	public boolean hasValue();

	/**
	 * Returns the inside of the element as text. <code>mytag-text value&lt;subtags /&gt;</code>.
	 * @since 1.1
	 * @return the inside of the element as text; <code>null</code> if the element did not contain any text
	 */
	public String getContent();

	/**
	 * Returns the name of the element. <tt>&lt;mytag-name&gt;mytag-text&lt;/mytag-name&gt;</tt>.
	 * If the element has a namespace prefix, eg. <tt>&lt;form:field /&gt;</tt>, the result will be field.
	 * @return the name of the element.
	 */
	public String getName();

	/**
	 * returns if there are more childnodes available.
	 * @since 1.4
	 * @return if there is a next childnode available. A childnode doesn't have to be a XmlTag,
	 * but can be any implementation of XmlNode
	 * @see XmlNode
	 */
	public boolean hasNextNode();

	/**
	 * returns the first available XmlNode that this tag contains.
	 * @return the first available XmlNode that this tag contains.
	 * @since 1.4
	 */
	public XmlNode getNextNode();

	/**
	 * Resets all browsepositions
	 * @since 1.5
	 */
	public void reset();

	/**
	 * Returns the number of tags available with this tagname
	 * @param tagName the tagname to count
	 * @return the number of tags available with this tagname
	 */
	public int countTags(String tagName);

	/**
	 * Returns the namespace URI.
	 * @since 1.6
	 * @return the namespace URI. If no namespace is specified, <tt>null</tt> is returned.
	 */
	public String getNamespaceURI();

	/**
	 * Locks the retrieval of 'next' tags to the namespace of this element.
	 * @since 1.6
	 * @param lock
	 */
	public void setNamespaceLock(boolean lock);

}