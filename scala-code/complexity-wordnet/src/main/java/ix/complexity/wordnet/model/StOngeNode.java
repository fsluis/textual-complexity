package ix.complexity.wordnet.model;

import edu.mit.jwi.item.IPointer;
import edu.mit.jwi.item.ISynsetID;
import edu.mit.jwi.item.Pointer;
import ix.common.util.ListUtil;

import java.io.IOException;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 13-apr-2010
 * Time: 19:35:49
 * To change this template use File | Settings | File Templates.
 */
public class StOngeNode extends Node {
    public static final int UP = 1, DOWN = 2, HORIZONTAL = 3, UNKNOWN = 0;
    public static final int[][] PATTERNS =
            {{UP}, {UP, DOWN}, {UP, HORIZONTAL}, {UP, HORIZONTAL, DOWN}, {DOWN}, {DOWN, HORIZONTAL}, {HORIZONTAL, DOWN}, {HORIZONTAL}};
    // patterns is based on stonge1995 p. 20 fig. 2.7b, "allowed patterns"
    public static final IPointer[] HORIZONTAL_POINTERS =
            {Pointer.ALSO_SEE, Pointer.ANTONYM, Pointer.ATTRIBUTE, Pointer.HOLONYM_MEMBER, Pointer.HOLONYM_PART, Pointer.HOLONYM_SUBSTANCE, Pointer.PERTAINYM, Pointer.SIMILAR_TO};
    public static final IPointer[] DOWN_POINTERS =
            {Pointer.CAUSE, Pointer.ENTAILMENT, Pointer.HYPONYM, Pointer.HYPONYM_INSTANCE};
    public static final IPointer[] UP_POINTERS =
            {Pointer.HYPERNYM, Pointer.HYPERNYM_INSTANCE, Pointer.MERONYM_MEMBER, Pointer.MERONYM_PART, Pointer.MERONYM_SUBSTANCE};
    public static final double C = 6.5, k = .5;

    private int history = UNKNOWN;

    public StOngeNode(SortedSet<ISynsetID> synsets, Node parent, int history) {
        super(synsets, parent);
        this.history = history;
    }

    public StOngeNode() throws IOException {
        super();
    }

    public List<Node> parse(WordNetUtil2 wordnet) {
        //long start = System.currentTimeMillis();
        List<Node> children = new Vector<Node>();

        for (int direction : getDirections()) {
            Set<IPointer> pointers = getPointers(direction);
            SortedSet<ISynsetID> related = new TreeSet<ISynsetID>(new SynsetComparator());
            for (IPointer pointer : pointers)
                related.addAll(wordnet.getRelatedSynsets(getSynsets(), pointer));
            related.removeAll(getAllSynsets());
            if (related.size() > 0) {
                children.add(new StOngeNode(related, this, direction));
            }
        }

        return children;
    }

    public double computeDistance() {
        //return C - getDepth() - k * getHistoryPattern().length;
        return getDepth() + k * getHistoryPattern().length;
    }

    private List<Integer> getHistory() {
        List<Integer> history = new LinkedList<Integer>();
        if (hasParent())
            history.addAll(((StOngeNode)getParent()).getHistory());
        history.add(this.history);
        return history;
    }

    public static int getDirection(IPointer pointer) {
        if (ListUtil.contains(UP_POINTERS, pointer))
            return UP;
        else if (ListUtil.contains(DOWN_POINTERS, pointer))
            return DOWN;
        else if (ListUtil.contains(HORIZONTAL_POINTERS, pointer))
            return HORIZONTAL;
        System.out.println("Help! Uknown pointer " + pointer.getName() + "; " + pointer.getSymbol());
        return UNKNOWN;
    }

    public static Set<IPointer> getPointers(int direction) {
        switch (direction) {
            case UP:
                return new HashSet<IPointer>(Arrays.asList(UP_POINTERS));
            case DOWN:
                return new HashSet<IPointer>(Arrays.asList(DOWN_POINTERS));
            case HORIZONTAL:
                return new HashSet<IPointer>(Arrays.asList(HORIZONTAL_POINTERS));
            default:
                return new HashSet<IPointer>();
        }
    }

    public static Set<IPointer> getPointers(Set<Integer> directions) {
        Set<IPointer> pointers = new HashSet<IPointer>();
        for (int direction : directions)
            pointers.addAll(getPointers(direction));
        return pointers;
    }

    private List<int[]> matchPatterns(int[] history) {
        List<int[]> matches = new Vector<int[]>();
        for (int[] pattern : PATTERNS) {
            boolean equals = true;
            if (history.length > pattern.length)
                equals = false;
            else for (int i = 0; i < pattern.length; i++)
                if (i < history.length)
                    if (history[i] != pattern[i])
                        equals = false;
            if (equals)
                matches.add(pattern);
        }
        return matches;
    }

    private int[] getHistoryPattern() {
        List<Integer> pattern = new LinkedList<Integer>();
        int lastDirection = UNKNOWN;
        for (int direction : getHistory()) {
            if (direction != lastDirection) {
                pattern.add(direction);
                lastDirection = direction;
            }
        }
        return ListUtil.intsToArray(pattern);
    }

    private Set<Integer> getDirections() {
        Set<Integer> directions = new HashSet<Integer>();
        int[] history = getHistoryPattern();
        List<int[]> patterns = matchPatterns(history);
        for (int[] pattern : patterns)
            if (pattern.length > history.length)
                directions.add(pattern[history.length]);
            else
                directions.add(pattern[history.length - 1]);
        return directions;
    }

    public int hashCode() {
        int hash = 0;
        List<Integer> history = getHistory();
        for (int i=0; i<history.size(); i++)
            hash += (i+1)^10*history.get(i);
        return hash;
    }
}
