package ix.complexity.model;

import ix.common.util.DefaultComparator;

import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
* Created by IntelliJ IDEA.
* User: Frans
* Date: 22-Dec-2010
* Time: 15:35:16
* To change this template use File | Settings | File Templates.
*/
public class NToken<T> implements Comparable<NToken<T>> {
    private List<T> tokens;
    private Comparator<T> tokenComparator = new DefaultComparator<T>();

    public Comparator<T> getTokenComparator() {
        return tokenComparator;
    }

    public void setTokenComparator(Comparator<T> tokenComparator) {
        this.tokenComparator = tokenComparator;
    }

    public NToken(List<T> tokens) {
        //this.tokens = tokens;
        this.tokens = new LinkedList<T>(tokens);
    }

    public NToken() {
        this.tokens = new LinkedList<T>();
    }

    public NToken(T token, Set<T> tokens) {
        this.tokens = new LinkedList<T>();
        this.tokens.add(token);
        this.tokens.addAll(tokens);
    }

    public NToken(T token, Set<T> tokens, Comparator<T> tokenComparator) {
        this.tokens = new LinkedList<T>();
        this.tokens.add(token);
        this.tokens.addAll(tokens);
        this.tokenComparator = tokenComparator;
    }

    public NToken(T tokenA, Comparator<T> tokenComparator) {
        this.tokens = new LinkedList<T>();
        this.tokens.add(tokenA);
        this.tokenComparator = tokenComparator;
    }

    public NToken(T tokenA, T tokenB, Comparator<T> tokenComparator) {
        this.tokens = new LinkedList<T>();
        this.tokens.add(tokenA);
        this.tokens.add(tokenB);
        this.tokenComparator = tokenComparator;
    }

    public NToken(List<T> tokens, Comparator<T> tokenComparator) {
        this.tokens = new LinkedList<T>(tokens);
        this.tokenComparator = tokenComparator;
    }

    public List<T> getTokens() {
        return tokens;
    }

    protected void setTokens(List<T> tokens) {
        this.tokens = tokens;
    }

    public T getToken(int i) {
        return tokens.get(i);
    }

    public void addToken(T token) {
        tokens.add(token);
    }

    public int getNrOfTokens() {
        return tokens.size();
    }

    public int compareTo(NToken<T> token) {
        int result = new Integer(getNrOfTokens()).compareTo(token.getNrOfTokens());
        if ( (result==0) && (getNrOfTokens()>0) )
            return compareTo(token, 0);
        return result;
    }

    private int compareTo(NToken<T> token, int n) {
        int result = getTokenComparator().compare(getToken(n), token.getToken(n));
        //int result = tokens.get(n).compareTo(token.getToken(n));
        if ((result == 0) && (++n < getNrOfTokens()))
            return compareTo(token, n);
        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof NToken)
            return compareTo((NToken) o) == 0;
        return super.equals(o);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public NToken<T> shorten(int n) {
        NToken<T> token = new NToken<T>();
        token.setTokens(tokens.subList(0, tokens.size()-n));
        return token;
    }

    public T getLast() {
        return tokens.get(tokens.size()-1);
    }

    public T getFirst() {
        return tokens.get(0);
    }

    public static class SizeIndifferentNTokenComparator<T> implements Comparator<NToken<T>> {
        public int compare(NToken<T> a, NToken<T> b) {
            Comparator<T> comparator = a.getTokenComparator();
            for (int i=0; i<a.getNrOfTokens() && i<b.getNrOfTokens(); i++) {
                int result = comparator.compare(a.getToken(i), b.getToken(i));
                if (result!=0)
                    return result;
            }
            return 0;
        }
    }
}
