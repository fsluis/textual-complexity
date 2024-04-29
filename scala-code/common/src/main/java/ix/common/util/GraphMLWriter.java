package ix.common.util;

import java.io.*;
// SAX classes.
import org.xml.sax.*;
import org.xml.sax.helpers.*;
//JAXP 1.1
import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.stream.*;
import javax.xml.transform.sax.*;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 19-Nov-2010
 * Time: 12:32:33
 * To change this template use File | Settings | File Templates.
 */
public class GraphMLWriter {
    private TransformerHandler handler;
    private AttributesImpl atts = new AttributesImpl();

    public GraphMLWriter() {
    }

    public void header() throws TransformerConfigurationException, SAXException, FileNotFoundException {
        PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream("C:/Users/Frans/Desktop/graph.xml")));
        StreamResult streamResult = new StreamResult(out);
        SAXTransformerFactory tf = (SAXTransformerFactory) SAXTransformerFactory.newInstance();
        // SAX2.0 ContentHandler.
        handler = tf.newTransformerHandler();
        Transformer serializer = handler.getTransformer();
        serializer.setOutputProperty(OutputKeys.ENCODING,"UTF-8");
        //serializer.setOutputProperty(OutputKeys.DOCTYPE_SYSTEM,"users.dtd");
        serializer.setOutputProperty(OutputKeys.INDENT,"yes");
        handler.setResult(streamResult);
        handler.startDocument();
        //<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        // xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
        atts.addAttribute("", "", "xmlns", "CDATA", "http://graphml.graphdrawing.org/xmlns");
        //atts.addAttribute("", "", "xmnls:xsi", "CDATA", "http://www.w3.org/2001/XMLSchema-instance");
        //atts.addAttribute("", "", "xsi:schemaLocation", "CDATA", "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd");
        handler.startElement("","","graphml",atts);
    }

    public void footer() throws SAXException {
        handler.endElement("","","graphml");
        handler.endDocument();
    }

    public void addEdgeWeights() throws SAXException {
        //<key id="d1" for="edge" attr.name="weight" attr.type="double"/>
        atts.clear();
        atts.addAttribute("", "", "id", "CDATA", "d1");
        atts.addAttribute("", "", "for", "CDATA", "edge");
        atts.addAttribute("", "", "attr.name", "CDATA", "Edge Weight");
        atts.addAttribute("", "", "attr.type", "CDATA", "double");
        handler.startElement("", "", "key", atts);
        handler.endElement("", "", "key");
    }

    public void addNodeLabel() throws SAXException {
        atts.clear();
        atts.addAttribute("", "", "id", "CDATA", "d0");
        atts.addAttribute("", "", "for", "CDATA", "node");
        atts.addAttribute("", "", "attr.name", "CDATA", "Label");
        atts.addAttribute("", "", "attr.type", "CDATA", "string");
        handler.startElement("", "", "key", atts);
        handler.endElement("", "", "key");
    }

    public void beginGraph(String id) throws SAXException {
        // <graph id="G" edgedefault="undirected">
        atts.clear();
        atts.addAttribute("", "", "id", "CDATA", id);
        atts.addAttribute("", "", "edgedefault", "CDATA", "undirected");
        handler.startElement("","","graph", atts);
    }

    public void endGraph() throws SAXException {
        handler.endElement("","","graph");
    }

    public void addNode(String id, String label) throws SAXException {
        //<node id="n0">
        atts.clear();
        atts.addAttribute("", "", "id", "CDATA", id);
        handler.startElement("","","node", atts);
        atts.clear();
        atts.addAttribute("", "", "key", "CDATA", "d0");
        handler.startElement("","","data", atts);
        handler.characters(label.toCharArray(), 0, label.length());
        handler.endElement("", "", "data");
        handler.endElement("", "", "node");
    }

    public void addEdge(String id, String source, String target, String weight) throws SAXException {
        //<edge id="e0" source="n0" target="n2">
        //<data key="d1">1.0</data>
        //</edge>
        atts.clear();
        atts.addAttribute("", "", "id", "CDATA", id);
        atts.addAttribute("", "", "source", "CDATA", source);
        atts.addAttribute("", "", "target", "CDATA", target);
        handler.startElement("","","edge", atts);
        atts.clear();
        atts.addAttribute("", "", "key", "CDATA", "d1");
        handler.startElement("","","data", atts);
        handler.characters(weight.toCharArray(), 0, weight.length());
        handler.endElement("", "", "data");
        handler.endElement("", "", "edge");
    }
}
