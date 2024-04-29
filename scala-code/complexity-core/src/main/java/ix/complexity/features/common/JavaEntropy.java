package ix.complexity.features.common;

import com.google.common.collect.Multiset;
import com.google.common.collect.TreeMultiset;
import ix.complexity.model.NToken;
import ix.common.util.DefaultComparator;
import ix.common.util.MatrixVector;
import no.uib.cipr.matrix.VectorEntry;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import static ix.common.util.MathUtil.log2;

/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 1/22/14
 * Time: 3:22 PM
 * To change this template use File | Settings | File Templates.
 */
public class JavaEntropy {

    // From
    public static double getEntropyB(MatrixVector P, MatrixVector F) {
        double entropy = 0;
        double totalEntropy = 0;
        double totalFrequencies = 0;
        double size = 0;
        for(int i=0; i<F.size(); i++)
            size += F.get(i);
        for(int i=0; i<P.size(); i++) {
            double p = P.get(i);
            double f = F.get(i);
            if(p>0)
                entropy -= f/size * log2(p);
            totalEntropy += p;
            totalFrequencies += f/size;
        }
        return entropy;
    }

    public static double getEntropyA(MatrixVector P) {
        Iterator<VectorEntry> chances = P.iterator();
        double entropy = 0;
        double total = 0;
        while(chances.hasNext()) {
            double p = chances.next().get();
            if(p>0)
                entropy -= p * log2(p);
            total += p;
        }
        return entropy;
    }

    // -------------- From EntropyBean.java ---------------------------------------
    public static <T> Entropy getEntropy(T[] signal, int n) {
        return getEntropy(Arrays.asList(signal), n);
    }

    public static <T> Entropy getEntropy(List<T> tokens, int n) {
        return getEntropy(tokens,n, new DefaultComparator<T>());
    }

    public static <T> Entropy getEntropy(List<T> tokens, int n, Comparator<T> comparator) {
        if (n == 1)
            return calculateEntropy(tokens, comparator);
        return calculateEntropy(tokens, n);
    }

    // A bag with distinct tokens and their frequencies
    public static <T> Multiset<T> encodeTokens(List<T> tokens, Comparator<T> comparator) {
        Multiset<T> bag = TreeMultiset.create(comparator);
        bag.addAll(tokens);
        return bag;
    }


    public static <T> Multiset<NToken<T>> encodeNTokens(List<T> tokens, int n, boolean includeLast) {
        return encodeNTokens(tokens,n,includeLast,new DefaultComparator<T>());
    }

    // Combine consecutive size tokens to one token and put them in a multiset
    public static <T> Multiset<NToken<T>> encodeNTokens(List<T> tokens, int n, boolean includeLast, Comparator<T> comparator) {
        Multiset<NToken<T>> nTokens = TreeMultiset.create();
        for (int i = 0; i < tokens.size() - n + (includeLast ? 1 : 0); i++)
            nTokens.add(new NToken<T>(tokens.subList(i, i + n), comparator));
        return nTokens;
    }

    public static <T> Multiset<NToken<T>> encodeNTokensStatic(List<T> tokens, int n, boolean includeLast) {
        Multiset<NToken<T>> nTokens = TreeMultiset.create();
        for (int i = 0; i < tokens.size() - n + (includeLast ? 1 : 0); i++)
            nTokens.add(new NToken<T>(tokens.subList(i, i + n)));
        return nTokens;
    }

    public static <T extends Comparable> Entropy calculateEntropy(List<T> tokens) {
        return calculateEntropy(tokens, new DefaultComparator<T>());
    }

    public static <T> Entropy calculateEntropy(List<T> tokens, Comparator<T> comparator) {
        return         calculateEntropy(encodeTokens(tokens,comparator));
    }

    public static <T> Entropy calculateEntropy(Multiset<T> bag) {
        //System.out.println(bag.elementSet().toArray());
        Double result = 0.0;
        Double Ps;
        for (Object token : bag.elementSet()) {
            Ps = (double) bag.count(token) / bag.size(); //tokens.size();
            result -= Ps * log2(Ps);
        }
        return new Entropy(result, bag.elementSet().size());
    }

    private static <T> Entropy calculateEntropy(List<T> tokens, int N) {
        Multiset<NToken<T>> Bs = encodeNTokens(tokens, N);
        Multiset<NToken<T>> B = encodeNTokens(tokens, N-1, false);
        Double result = 0.0;
        double weights = 0;
        for (NToken bs : Bs.elementSet() ) {
            NToken b = bs.shorten(1);
            //double Pb = (double) B.count(b) / tokens.size();
            double Pb = (double) B.count(b) / B.size();
            double Pbs = (double) Bs.count(bs) / B.count(b);
            weights += Pb*Pbs;
            result  -= Pb * Pbs * log2(Pbs);
        }
        return new Entropy(result, Bs.elementSet().size());  //I don't know if this is the right size... maybe it should be Bs.size?? Or B.size?? :S It does look to be right (Bs.elementSet.size()): the maximum perplexity is dependent on the number of "tokens", where each token has a chance assigned to it. So, not the numbers of tokens in a set, but the number of unique tokens in a set.
    }

    private static <T> Multiset<NToken<T>> encodeNTokens(List<T> tokens, int n) {
        return encodeNTokens(tokens,n,true);
    }

    public static double calculateFacticity(double h, double hMax) {
        return 4*h/hMax*(hMax-h)/hMax;
    }

    public static double calculatePerplexity(double h) {
        return Math.pow(2, h);
    }

    public static double calculateNormalizedPerplexity(double h, int n) {
        return Math.pow(2, h)/n;
    }

    public static double calculateEntropyRatio(double h, int n) {
        return h / log2(n);
    }

    public static <T> Entropy calculateCrossEntropy(Multiset<T> test, Multiset<T> train) {
        //System.out.println(bag.elementSet().toArray());
        Double result = 0.0;
        Double Ps, Qs;
        for (Object token : test.elementSet()) {
            Ps = (double) test.count(token) / test.size();
            Qs = (double) train.count(token) / train.size();
            result -= Ps * log2(Qs);
        }
        return new Entropy(result, test.elementSet().size());
    }
}
