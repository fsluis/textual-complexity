package ix.common.core.xml;

import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * NodeSerializer is a tool to convert a <code>org.w3c.dom.Node</code> back to textform.
 * This is usefull if the Node contains any text that need to be parsed in another way (eg. by String replacement)
 *
 * @author m
 * @version 1.0
 */
public class NodeSerializer {

	/**
	 * This is the main function of this class. It converts a XML document Node from the Xerces package back into a String,
	 * so that it can be evaluated in a different way. The Childs of the Node are also parsed back into the String.
	 *
	 * @param input the XML Node that must be converted back into a String.
	 * @param include boolean if the tag itself must be included in the serialization process.
	 * @return the String representation of this XML node and its childs.
	 */
	public static String nodeToString(Node input, boolean include) {
		return nodeToString(input, 0, include);
	}

	private static String nodeToString(Node input, int level, boolean include) {
		if (input.getNodeType() == Node.TEXT_NODE) {
			return input.getNodeValue();
		}
		if ((input.getNodeType() == Node.ELEMENT_NODE) && ((level > 0) || (include))) {
			String tagName = input.getNodeName();
			String spacing = "";
			//for (int i = 0; i < level; i++) spacing += "\t";
			String output = spacing + "<" + tagName;
			if (input.hasAttributes()) {
				NamedNodeMap attributes = input.getAttributes();
				for (int i = 0; i < attributes.getLength(); i++) {
					Node attribute = attributes.item(i);
					output += " " + attribute.getNodeName() + "=\"" + attribute.getNodeValue() + "\"";
				}
			}
			if (input.hasChildNodes()) {
				output += ">";
				NodeList subTags = input.getChildNodes();
				for (int i = 0; i < subTags.getLength(); i++) {
					Node subTag = subTags.item(i);
					output += nodeToString(subTag, level + 1, include);
				}
				output += spacing + "</" + tagName + ">";
			} else
				output += " />";
			return output;
		}
		if ((input.getNodeType() == Node.ELEMENT_NODE) && (level == 0) && (!include)) {
			String spacing = "";
			//for (int i = 0; i < level; i++) spacing += "\t";
			String output = spacing;
			if (input.hasChildNodes()) {
				NodeList subTags = input.getChildNodes();
				for (int i = 0; i < subTags.getLength(); i++) {
					Node subTag = subTags.item(i);
					output += nodeToString(subTag, level + 1, include);
				}
			}
			return output;
		}
		return "";
	}

}