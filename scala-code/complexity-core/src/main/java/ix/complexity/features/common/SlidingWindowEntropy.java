package ix.complexity.features.common;

import com.google.common.collect.Multiset;
import com.google.common.collect.TreeMultiset;
import ix.complexity.model.NToken;
import ix.common.util.LimitedCapacityQueue;
import ix.common.util.TimeLogger;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.lang.reflect.Array;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 4/7/12
 * Time: 11:54 AM
 * To change this template use File | Settings | File Templates.
 */
public class SlidingWindowEntropy {
    private static final Log log = LogFactory.getLog(SlidingWindowEntropy.class);
    private static final TimeLogger tlog = new TimeLogger(log, "sliding_window_entropy");

    /**
     * Calculate entropy of sequences of symbols over a window
     * @param tokens list of symbols
     * @param n what sequences of symbols to evaluate (1...n)
     * @param capacity the window size
     * @return a summary statistic of the symbol entropy for each sequence 1...n
     */
    public static <T> DescriptiveStatistics[] calculateEntropy(List<T> tokens, int n, int capacity) {
        long time = tlog.start();
        // Prepare stats
        DescriptiveStatistics stats[] = new DescriptiveStatistics[n];
        for(int i=0; i<n; i++)
            stats[i] = new DescriptiveStatistics();

        // Check reqs
        if (tokens.size()<=capacity)
            return stats;

        // Prepare cache
        EntropyCache<T> cache = new EntropyCache<T>(n, capacity);
        cache.refill(tokens.subList(0,capacity));

        // Calc entropy windows
        for(int i=capacity; i<tokens.size(); i++) {
            for(int j=0; j<n; j++)
                stats[j].addValue(cache.calculateEntropy(j));
            cache.add(tokens.get(i));
        }

        // Resuls
        /*double entropies[] = new double[n];
        for(int i=0; i<n; i++)
            entropies[i] = stats[i].getMean();
        return entropies;*/
        tlog.stop(time);
        return stats;
    }

    static class EntropyCache<T> {
        Multiset<NToken<T>>[] ntokens;
        LimitedCapacityQueue<T> tokens;
        private int N, capacity;
        public static final boolean includeLast=true;

        private EntropyCache(int n, int capacity) {
            N = n;
            this.capacity = capacity;
            tokens = new LimitedCapacityQueue<T>(capacity);
            ntokens = (Multiset<NToken<T>>[]) Array.newInstance(Multiset.class, N);
            for(int i=0;i<N;i++)
                ntokens[i] = TreeMultiset.create();
        }

        public void refill(List<T> tokens) {
            //refill tokens
            this.tokens.clear();
            this.tokens.addAll(tokens);

            //refill ntokens
            for(int i=0;i<N;i++) {
                ntokens[i].clear();
                fill(ntokens[i], tokens, i+1);
            }
        }

        // Combine consecutive size tokens to one token and put them in a multiset
        private void fill(Multiset<NToken<T>> nTokens, List<T> tokens, int n) {
            for (int i = 0; i < tokens.size() - n + (includeLast ? 1 : 0); i++)
                nTokens.add(new NToken<T>(tokens.subList(i, i + n)));
        }

        public T add(T tail) {
            // Remove stats for head
            for(int i=0;i<N;i++) {
                NToken<T> ntoken = new NToken<T>(tokens.subList(0, i + 1));
                ntokens[i].remove(ntoken);
            }
            // Remove head, append tail
            T head = tokens.append(tail);
            // Add tail
            //tokens.add(tail);
            // Add stats for tail
            int size = tokens.size();
            size = capacity;
            for(int i=0;i<N;i++) {
                NToken<T> ntoken = new NToken<T>(tokens.subList(size-i-1, size));
                ntokens[i].add(ntoken);
            }
            return head;
        }

        public double calculateEntropy(int n) {
            return JavaEntropy.calculateEntropy(ntokens[n]).entropy();
        }
    }

}
