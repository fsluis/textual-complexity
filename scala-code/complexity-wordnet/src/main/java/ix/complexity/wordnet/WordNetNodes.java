package ix.complexity.wordnet;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.collect.TreeMultiset;
import edu.mit.jwi.item.ISynsetID;
import edu.mit.jwi.item.POS;
import ix.complexity.wordnet.model.Node;
import ix.complexity.wordnet.model.WordNetUtil2;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.net.MalformedURLException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.SortedSet;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 3/31/11
 * Time: 6:25 PM
 * To change this template use File | Settings | File Templates.
 */
public class WordNetNodes {
    private static final Log log = LogFactory.getLog(WordNetNodes.class);
    private List<POS> poses = Arrays.asList(POS.NOUN, POS.VERB);
    private Class<? extends Node> nodeType = Node.class;

    public Multiset<Node> doReduce(List<String> tokens, WordNetUtil2 wordnet) throws MalformedURLException, IllegalAccessException, InstantiationException {
        Multiset<String> words = TreeMultiset.create();
        Multiset<Node> nodes = HashMultiset.create();
        words.addAll(tokens);
        Iterator<Multiset.Entry<String>> iterator = words.entrySet().iterator();
        long n = 0;
        while (iterator.hasNext()) {
            Multiset.Entry<String> word = iterator.next();
            SortedSet<ISynsetID> synsets = wordnet.findSynsets(word.getElement(), poses);
            Node node = nodeType.newInstance();
            node.setSynsets(synsets);
            nodes.add(node, word.getCount());
            //for(int i=0; i<word.getCount(); i++)
            //    collector.collect(n++, node);
        }
        return nodes;
    }

    public List<POS> getPoses() {
        return poses;
    }

    public void setPoses(List<POS> poses) {
        this.poses = poses;
    }

    public Class<? extends Node> getNodeType() {
        return nodeType;
    }

    public void setNodeType(Class<? extends Node> nodeType) {
        this.nodeType = nodeType;
    }
}
