package ix.common.core.xml;

import org.w3c.dom.*;
import org.xml.sax.InputSource;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;

/**
 * Document is a parsed XML Document. It is an extra layer over Xerces, to make the parsing process of an XML document a lot easier.
 *
 * An example of an XML document:
 * <code><pre>
 * &lt;person&gt;
 *   &lt;hobbies&gt;
 *     &lt;hobby&gt;Tennis&lt;/hobby&gt;
 *     &lt;hobby favorite="yes"&gt;Java&lt;/hobby&gt;
 *     &lt;hobby&gt;Computers&lt;/hobby&gt;
 *   &lt;/hobbies&gt;
 * &lt;/person&gt;
 * </pre></code>
 * A typical way of parsing a document is as follows:
 * <code><pre>
 * Document d = new Document("foobar.xml");
 * if (d.hasNextTag("hobbies")) {
 *   XmlTag hobbies = d.getNextTag("hobbies");
 *   while(hobbies.hasNextTag("hobby")) {
 *     XmlTag hobby = hobbies.getNextTag("hobby");
 *     String hobbyName = hobby.getValue();
 *     boolean isFavorite = hobby.getAttributeValue("favorite", "no").equals("yes");
 *   }
 * }
 * </pre></code>
 *
 * @author m
 * @version 1.4
 */
public class Document implements XmlTag {

	private HashMap tagIndices = new HashMap();
	// This HashMap is needed to store the positions of the tags in.
	// This is nessecary because we need to tell if there are more tags available with the same
	// name. therefor for each tagname we store the "current" position.

	private int childsReturned = 0;
	// Keep the index of the number of elements that were returned.
	// for each different element another index is stored in the tagIndices HashMap,
	// to be able to search the next of each kind.

	private Element element;
	private boolean namespaceLock = false;
	// The Xerces tag we are representing with this XmlTag element.

	/**
	 * Parse the content of the given file as an XML document.
	 * @param f The file containing the XML to parse.
	 * @exception java.io.IOException If any IO errors occur.
	 * @exception XmlParseException If any parse errors occur.
	 * @exception IllegalArgumentException If the file is null.
	 */
	public Document(java.io.File f)
		throws XmlParseException, java.io.IOException {
		try {
			this.element = getDocumentBuilder().parse(f).getDocumentElement();
		} catch(javax.xml.parsers.ParserConfigurationException pce) {
			new XmlParseException(pce);
		} catch(org.xml.sax.SAXException se) {
			new XmlParseException(se);
		}
	}

	/**
	 * Parse the content of the given input source as an XML document.
	 * @param is InputSource containing the content to be parsed.
	 * @exception java.io.IOException If any IO errors occur.
	 * @exception XmlParseException If any parse errors occur.
	 * @exception IllegalArgumentException If the InputSource is null.
	 */
	public Document(InputSource is)
		throws XmlParseException, java.io.IOException {
		try {
			this.element = getDocumentBuilder().parse(is).getDocumentElement();
		} catch(javax.xml.parsers.ParserConfigurationException pce) {
			new XmlParseException(pce);
		} catch(org.xml.sax.SAXException se) {
			new XmlParseException(se);
		}
	}

	/**
	 * Parse the content of the given <code>InputStream</code> as an XML
	 * document.
	 * @param is InputStream containing the content to be parsed.
	 * @exception java.io.IOException If any IO errors occur.
	 * @exception XmlParseException If any parse errors occur.
	 * @exception IllegalArgumentException If the InputStream is null
	 */
	public Document(java.io.InputStream is)
		throws XmlParseException, java.io.IOException {
		try {
			this.element = getDocumentBuilder().parse(is).getDocumentElement();
		} catch(javax.xml.parsers.ParserConfigurationException pce) {
			new XmlParseException(pce);
		} catch(org.xml.sax.SAXException se) {
			new XmlParseException(se);
		}
	}

	/**
	 * Parse the content of the given <code>InputStream</code> as an XML
	 * document.
	 * @param is InputStream containing the content to be parsed.
	 * @param systemId Provide a base for resolving relative URIs.
	 * @exception java.io.IOException If any IO errors occur.
	 * @exception XmlParseException If any parse errors occur.
	 * @exception IllegalArgumentException If the InputStream is null.
	 */
	public Document(java.io.InputStream is, java.lang.String systemId)
		throws XmlParseException, java.io.IOException {
		try {
			this.element = getDocumentBuilder().parse(is, systemId).getDocumentElement();
		} catch(javax.xml.parsers.ParserConfigurationException pce) {
			new XmlParseException(pce);
		} catch(org.xml.sax.SAXException se) {
			new XmlParseException(se);
		}
	}



	/**
	 * Parse the content of the given URI as an XML document.
	 * @param uri The location of the content to be parsed.
	 * @exception java.io.IOException If any IO errors occur.
	 * @exception XmlParseException If any parse errors occur.
	 * @exception IllegalArgumentException If the URI is null.
	 */
	public Document(java.lang.String uri)
		throws XmlParseException, java.io.IOException {
		try {
			this.element = getDocumentBuilder().parse(uri).getDocumentElement();
		} catch(javax.xml.parsers.ParserConfigurationException pce) {
			new XmlParseException(pce);
		} catch(org.xml.sax.SAXException se) {
			new XmlParseException(se);
		}
	}

	/**
	 * Create a new Document using a <code>org.w3c.dom.Document</code> as source.
	 * @param document The <code>org.w3c.dom.Document</code> to use as source.
	 */
	public Document(org.w3c.dom.Document document) {
		this.element = document.getDocumentElement();
	}

	/**
	 * This constructor is to allow Document to create other Document objects and let them function as XMLTags.
	 *@param element the Xerces xml element, XmlTag is the wrapper for this element
	 */
	private Document(Element element) {
		this.element = element;
	}

	/**
	 * Sets the configuration for the Xerces document builder and returns the builder.
	 * This is to have faster access to the builder, and to set the configuration on a central place.
	 *@return DocumentBuilder the Xerces documentbuilder
	 *@throws javax.xml.parsers.ParserConfigurationException the ParserConfigurationException is thrown if an error occurs
	 * while configuring the DocumentBuilder
	 */
	private DocumentBuilder getDocumentBuilder()
		throws javax.xml.parsers.ParserConfigurationException {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		dbf.setValidating(false);
		dbf.setNamespaceAware(true);
		return dbf.newDocumentBuilder();
	}

	/**
	 * Checks wether there is a next element available with the tagname <code>tagname</code>.
	 * @param tagname the name of the tag to be checked on availability.
	 * @return <code>true</code> if there is a next tag available with the tagname <code>tagname</code>;
	 *         <code>false</code> otherwise.
	 * @since 1.3 order of method changed, should be faster now
	 */
	public boolean hasNextTag(String tagname) {
		// first recieve what the count of the tagname minimal has to be.
		int position = 0;
		if (tagIndices.containsKey(tagname)) {
			// if this tag was asked for we stored the new position in the Map.
			// retreive this position now to check if more tags are available.
			position = ((Integer) tagIndices.get(tagname)).intValue();
		}

		// iterate all children, and count the occurences of the tagname
		NodeList childs = element.getChildNodes();
		int length = 0;
		for (int i = 0; i < childs.getLength(); i++) {
			// Look through all child elements and check if the tagname is the one we are looking for
			Node child = childs.item(i);
			if (child.getNodeType() == Node.ELEMENT_NODE) {
				Element nextElement = (Element) child;
				if (nameNoNS(nextElement).equals(tagname)) {
					if (!namespaceLock)
						length++;
					else {
						String namespaceNow = element.getNamespaceURI();
						String nextNamespace = nextElement.getNamespaceURI();
						if (((namespaceNow == null) && (nextNamespace == null)) ||
							((namespaceNow != null) && (namespaceNow.equals(nextNamespace))))
							length++;
					}
				}
			}
			if (position < length) return true;
		}
		// else: This is not the tag you are looking for (waves hand jedi style)
		return false;
	}

	private String nameNoNS(Element xmlElement) {
		String prefix = xmlElement.getPrefix();
		String tagName = xmlElement.getTagName();
		if (prefix != null) tagName = tagName.substring(prefix.length() + 1);
		return tagName;
	}

	/**
	 * Returns the next available <code>XmlTag</code> with the tagname <code>tagname</code>
	 * @param tagname the name of the tag to be returned.
	 * @return the next available <code>XmlTag</code> with the tagname <code>tagname</code>, <code>null</code> if the tag was not available.
	 * @see	#hasNextTag(String)
	 */
	public XmlTag getNextTag(String tagname) {
		// first retreave all tags with this name in an List
		int position = 0;
		if (tagIndices.containsKey(tagname)) {
			// if this tag was asked for we stored the new position in the Map.
			// retreive this position now to check if more tags are available.
			position = ((Integer) tagIndices.get(tagname)).intValue();
		}

		NodeList childs = element.getChildNodes();
		int itemCount = 0;
		for (int i = 0; i < childs.getLength(); i++) {
			// Look through all child elements and check if the tagname is the one we are looking for
			Node child = childs.item(i);
			if (child.getNodeType() == Node.ELEMENT_NODE) {
				Element nextElement = (Element) child;
				if (nameNoNS(nextElement).equals(tagname)) {
					if (!namespaceLock) {
						if (itemCount == position) {
							tagIndices.put(tagname, new Integer(position + 1));
							return new Document((Element) child); // create a new Document for the XmlTag interface
						}
						itemCount++;
					}	else {
						String namespaceNow = element.getNamespaceURI();
						String nextNamespace = nextElement.getNamespaceURI();
						if (((namespaceNow == null) && (nextNamespace == null)) ||
							((namespaceNow != null) && (namespaceNow.equals(nextNamespace)))) {
							if (itemCount == position) {
								tagIndices.put(tagname, new Integer(position + 1));
								return new Document((Element) child); // create a new Document for the XmlTag interface
							}
							itemCount++;
						}
					}
				}
			}
			// else: This is not the tag you are looking for (waves hand jedi style)
		}
		return null;
	}

    public List<XmlTag> getTags(String tagName) {
        List<XmlTag> tags = new Vector<XmlTag>();
        NodeList nodes = element.getElementsByTagName(tagName);
        for(int i=0; i<nodes.getLength(); i++) {
            Node node = nodes.item(i);
            if(node.getNodeType()==Node.ELEMENT_NODE)
                tags.add(new Document((Element)node));
        }
        return tags;
    }

    /**
	 * config if the XmlTag has any subnodes. Subnodes are not only tags, but also pieces of text.
	 * @return true if there are more nodes to return, false otherwise.
	 */
	public boolean hasNextNode() {
		NodeList childs = element.getChildNodes();
		return (childsReturned < childs.getLength());
	}

	/**
	 * Returns a next node from the xmltag. When the node is of the type tag, it can safely be casted into
	 * an XmlTag. This method returns all nodes in the order they are encountered in the structure.
	 * @return an XmlNode (text or tag) that this tag contains. <code>null</code> is returned when there are
	 * no more nodes left.
	 */
	public XmlNode getNextNode() {
		NodeList childs = element.getChildNodes();
		Node child = childs.item(childsReturned);
		childsReturned++;
		if (child.getNodeType() == Node.ELEMENT_NODE) {
			return new Document((Element) child);
		}
		if (child.getNodeType() == Node.TEXT_NODE) {
			return new XmlTextNode(child);
		}
		if (child.getNodeType() == Node.COMMENT_NODE) {
			return new XmlCommentNode(child);
		}
		return null;
	}

	/**
	 * Resets all browsepositions
	 */
	public void reset() {
		tagIndices.clear();
    childsReturned = 0;
	}

	/**
	 * Returns the number of tags available with this tagname
	 * @param tagName the tagname to count
	 * @return the number of tags available with this tagname
	 */
	public int countTags(String tagName) {
		NodeList result = element.getElementsByTagName(tagName);
		return result.getLength();
	}

	/**
	 * Returns the namespace URI.
	 * @return the namespace URI. If no namespace is specified, <tt>null</tt> is returned.
	 */
	public String getNamespaceURI() {
		return element.getNamespaceURI();
	}

	/**
	 * Private class to implement the XmlNode interface for textnodes in the tags.
	 */
	private class XmlTextNode implements XmlNode {

		private Node textNode;

		public XmlTextNode(Node content) {
			this.textNode = content;
		}

		public int nodeType() {
			return TYPE_TEXT;
		}

		public String getContent() {
			return textNode.getNodeValue();
		}

        public String getName() {
            return textNode.getNodeName();
        }
	}

	private class XmlCommentNode implements XmlNode {

		private Node commentNode;

		public XmlCommentNode(Node content) {
			this.commentNode = content;
		}

		public int nodeType() {
			return TYPE_COMMENT;
		}

		public String getContent() {
			return commentNode.getNodeValue();
		}

        public String getName() {
            return commentNode.getNodeName();
        }
	}

	/**
	 * Returns the value of an attribute of a tag. Use the <code>defaultvalue</code> parameter to specify the returned value if the attribute was not found.
	 * @param attribute the name of the attribute to return the value of
	 * @param defaultvalue the value that will be returned if the attribute was not found
	 * @return the value of an attribute of a tag. Use the <code>defaultvalue</code> parameter to specify the returned value if the attribute was not found.
	 * @see	#getAttributeValue(String attribute)
	 */
	public String getAttributeValue(String attribute, String defaultvalue) {
		if (element.hasAttribute(attribute))
			return getAttributeValue(attribute);
		return defaultvalue;
	}

	/**
	 * Returns the value of an attribute of a tag. <code>attributename="attributevalue"</code>. To get this value, use getAttributeValue("attributename").
	 * @param attribute the name of the attribute to return the value of
	 * @return the value of an attribute of a tag. <code>attributename="attributevalue"</code>. To get this value, use getAttributeValue("attributename").
	 *         <code>null</code> is returned if the specified attribute was not found.
	 * @see	#getAttributeValue(String attribute, String defaultvalue)
	 */
	public String getAttributeValue(String attribute) {
		return element.getAttribute(attribute);
	}

	/**
	 * Returns a Map of attributes of a tag.
	 * @return a Map of attributes of a tag.
	 * @since 1.1
	 */
	public Map getAttributes() {
		Map result = new HashMap();
		NamedNodeMap attributes = element.getAttributes();
		for (int i = 0; i < attributes.getLength(); i++) {
			Node attribute = attributes.item(i);
			result.put(attribute.getNodeName(), attribute.getNodeValue());
		}
		return result;
	}

	/**
	 * Checks wether the attribute with the tagname <code>attribute</code> is available.
	 * @param attribute the name of the attribute to be checked on availability.
	 * @return <code>true</code> if the attribute is available
	 * @return <code>false</code> otherwise.
	 */
	public boolean hasAttributeValue(String attribute) {
		return element.hasAttribute(attribute);
	}

	/**
	 * Returns the textual value of the element. <code>&lt;mytag&gt;mytag-text value&lt;/mytag&gt;</code>.
	 * @return the textual value of the element; an empty <code>String</code> if the element did not contain any text
	 */
	public String getValue() {
		// this will only work succesfully if there is only text in the tagcontent
		Node text = element.getFirstChild();
		if ((text != null) && (text.getNodeType() == Node.TEXT_NODE)) {
			return text.getNodeValue();
		}
		return new String();
	}

	/**
	 * Returns wether this XmlTag has an value or not.
	 * @return <code>true</code> if the Tag has an textvalue, <code>false</code> otherwise
	 * @since 1.2
	 */
	public boolean hasValue() {
		Node text = element.getFirstChild();
		if ((text != null) && (text.getNodeType() == Node.TEXT_NODE)) {
			return true;
		}
		return false;
	}


	/**
	 * Returns the name of the element. <tt>&lt;mytag-name&gt;mytag-text&lt;/mytag-name&gt;</tt>.
	 * If the element has a namespace prefix, eg. <tt>&lt;form:field /&gt;</tt>, the result will be field.
	 * @return the name of the element.
	 */
	public String getName() {
		return nameNoNS(element);
	}

	public int nodeType() {
		return TYPE_TAG;
	}

	/**
	 * Returns the XML document as an String containing the XML source
	 */
	public String getContent() {
		return NodeSerializer.nodeToString(element, false);
	}

	/**
	 * Returns the XML document as an String containing the XML source
	 */
	public String toString() {
		return NodeSerializer.nodeToString(element, true);
	}

	/**
	 * Locks the retrieval of 'next' tags to the namespace of this element.
	 * @since 1.6
	 * @param lock
	 */
	public void setNamespaceLock(boolean lock) {
		this.namespaceLock = lock;
	}
}