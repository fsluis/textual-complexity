package ix.complexity.wordnet;

import ix.complexity.wordnet.model.Node;
import ix.complexity.wordnet.model.WordNetUtil2;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 4/5/11
 * Time: 1:12 PM
 * To change this template use File | Settings | File Templates.
 */
public class WordNetSpread {
    public static int MAXDEPTH = 5;

    public static Integer[][] doMap(Node node, Integer maxDepth, WordNetUtil2 wordnet) {
        Integer[][] spread = new Integer[2][maxDepth];
        computeSpread(node, 0, spread, maxDepth, wordnet);
        return spread;
    }

    public static void computeSpread(Node node, int depth, Integer[][] spread, int maxDepth, WordNetUtil2 wordnet) {
        spread[0][depth] = node.getSynsets().size();
        spread[1][depth] = node.getAllSynsets().size();
        if (depth + 1 < maxDepth)
            for (Node child : node.getChildren(wordnet))
                computeSpread(child, depth + 1, spread, maxDepth, wordnet);
    }
}
