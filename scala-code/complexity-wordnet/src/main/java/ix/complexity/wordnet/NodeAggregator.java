package ix.complexity.wordnet;

import ix.complexity.wordnet.model.Node;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.Iterator;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 4/5/11
 * Time: 2:27 PM
 * To change this template use File | Settings | File Templates.
 */
public class NodeAggregator {
    private static final Log log = LogFactory.getLog(NodeAggregator.class);
    private Class<? extends Node> nodeType = Node.class;

    public Node doReduce(Iterator<Node> nodes) throws IllegalAccessException, InstantiationException {
        Node total = nodeType.newInstance();

        while (nodes.hasNext())
            total.addSynsets(nodes.next().getSynsets());

        return total;
    }

    public Class<? extends Node> getNodeType() {
        return nodeType;
    }

    public void setNodeType(Class<? extends Node> nodeType) {
        this.nodeType = nodeType;
    }

}
