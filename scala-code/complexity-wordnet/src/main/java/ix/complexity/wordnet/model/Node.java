package ix.complexity.wordnet.model;

import edu.mit.jwi.item.ISynsetID;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 7-apr-2010
 * Time: 14:23:27
 * To change this template use File | Settings | File Templates.
 */
public class Node implements Serializable {
    private SortedSet<ISynsetID> synsets;
    private Node parent;
    private List<Node> children;

    public Node() throws IOException {
        synsets = new TreeSet<ISynsetID>(new SynsetComparator());
    }

    protected Node(SortedSet<ISynsetID> synsets, Node parent) {
        this.synsets = synsets;
        this.parent = parent;
    }

    public int getDepth() {
        if (hasParent())
            return getParent().getDepth()+1;
        return 0;
    }

    public List<Node> parse(WordNetUtil2 wordnet) {
        SortedSet<ISynsetID> synsets = new TreeSet<ISynsetID>(new SynsetComparator());
        for (ISynsetID synset : getSynsets())
            synsets.addAll(wordnet.getRelatedSynsetsWithoutHyponymy(synset));
        if (hasParent())
            synsets.removeAll(getParent().getAllSynsets());

        Node child = new Node(synsets, this);
        List<Node> children = new Vector<Node>();
        children.add(child);
        return children;
    }

    public SortedSet<ISynsetID> getAllSynsets(int minDepth, int maxDepth, WordNetUtil2 wordnet) {
        SortedSet<ISynsetID> synsets = new TreeSet<ISynsetID>(new SynsetComparator());
        if (minDepth<=getDepth())
            synsets.addAll(getSynsets());
        if (getDepth()<maxDepth)
            for (Node node : getChildren(wordnet))
                synsets.addAll(node.getAllSynsets(minDepth, maxDepth, wordnet));
        return synsets;
    }

    public SortedSet<ISynsetID> getAllSynsets() {
        SortedSet<ISynsetID> all = new TreeSet<ISynsetID>(new SynsetComparator());
        if (hasParent())
            all.addAll(getParent().getAllSynsets());
        all.addAll(synsets);
        return all;
    }

    public SortedSet<ISynsetID> getSynsets() {
        return synsets;
    }

    public void setSynsets(SortedSet<ISynsetID> synsets) {
        this.synsets = synsets;
    }

    public void addSynsets(SortedSet<ISynsetID> synsets) {
        this.synsets.addAll(synsets);
    }

    public Collection<Node> getChildren(WordNetUtil2 wordnet) {
        if (children == null)
            children = parse(wordnet);
        return children;
    }

    /**
     * This method returns, in no particular order, the nodes (siblings, parents, itself, etc.) which contain at least on of the needles.
     * @param needles
     * @param minDepth
     * @param maxDepth
     * @return
     */
    public List<Node> searchSynsets(Collection<ISynsetID> needles, int minDepth, int maxDepth, WordNetUtil2 wordnet) {
        int depth = getDepth();
        if (minDepth <= depth && depth < maxDepth)
            if (searchSynsets(needles))
                return Arrays.asList(this);

        List<Node> hay = new Vector<Node>();
        if (depth < maxDepth)
            for (Node node : getChildren(wordnet))
                hay.addAll(node.searchSynsets(needles, minDepth, maxDepth, wordnet));
        return hay;
    }

    public boolean searchSynsets(Collection<ISynsetID> needles) {
        return WordNetUtil2.intersection(synsets, needles).size() > 0;
    }

    public boolean hasParent() {
        return (parent != null);
    }

    public Node getParent() {
        return parent;
    }

    public double computeDistance() {
        return getDepth();
    }

    public int hashCode() {
        return getDepth();
    }
}
