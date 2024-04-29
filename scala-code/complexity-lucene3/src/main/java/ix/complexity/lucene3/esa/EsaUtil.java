package ix.complexity.lucene3.esa;

import gnu.trove.function.TDoubleFunction;
import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.procedure.TDoubleProcedure;
import gnu.trove.procedure.TIntDoubleProcedure;
import ix.common.util.Matrix;
import ix.common.util.MatrixVector;

import java.util.Collection;
import java.util.Iterator;
import java.util.Map;

import static ix.complexity.features.common.JavaEntropy.getEntropyA;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 12/13/11
 * Time: 1:00 AM
 * To change this template use File | Settings | File Templates.
 */
public class EsaUtil {
    public static Matrix cosineMatrix(Collection<TIntDoubleHashMap> concepts) {
        Matrix cosine = new Matrix(concepts.size(), concepts.size());
        int i=0, j=0;
        double numerator, denominator;
        for (TIntDoubleHashMap conceptA : concepts) {
            for (TIntDoubleHashMap conceptB : concepts) {
                cosine.set(i,j,cosine(conceptA,conceptB) );
                j++;
            }
            i++;
            j=0;
        }
        return cosine;
    }

    public static double cosine(TIntDoubleHashMap conceptA, TIntDoubleHashMap conceptB) {
        double numerator = dotProduct(conceptA, conceptB);
        double denominator = Math.sqrt(squaredNorm(conceptA)) * Math.sqrt(squaredNorm(conceptB));
        return numerator/denominator;
    }

    public static double dotProduct(Map<Integer, Double> A, Map<Integer, Double> B) {
        double product = 0, b;
        for(Map.Entry<Integer, Double> a : A.entrySet()) {
            b = B.get(a.getKey());
            product += a.getValue()*b;
        }
        return product;
    }

    public static double dotProduct(TIntDoubleHashMap A, TIntIntHashMap B) {
        double product = 0, b;
        TIntDoubleIterator iter = A.iterator();
        while(iter.hasNext()) {
            iter.advance();
            b = B.get(iter.key());
            product += iter.value()*b;
        }
        return product;
    }

    public static double dotProduct(TIntDoubleHashMap A, TIntDoubleHashMap B) {
        double product = 0, b;
        TIntDoubleIterator iter = A.iterator();
        while(iter.hasNext()) {
            iter.advance();
            b = B.get(iter.key());
            product += iter.value()*b;
        }
        return product;
    }

    public static double squaredNorm(Map<Integer, Double> A) {
        return dotProduct(A, A);
    }

    public static double squaredNorm(TIntDoubleHashMap A) {
        //return dotProduct(A, A);
        final double[] norm = {0};
        A.forEachValue(new TDoubleProcedure() {
            public boolean execute(double v) {
                norm[0] += v*v;
                return true;
            }
        });
        return norm[0];
    }

    /**
     * Performs A-B
     * @param A vector
     * @param B vector
     * @return clone of A, reduced with B
     */
    public static TIntDoubleHashMap min(TIntDoubleHashMap A, TIntDoubleHashMap B) {
        TIntDoubleHashMap C = new TIntDoubleHashMap(A);// See TroveCopyTest for prove that this performs a deep copy
        TIntDoubleIterator b = B.iterator();
        double value;
        while(b.hasNext()) {
            b.advance();
            value = -1 * b.value();
            C.adjustOrPutValue(b.key(), value, value);
        }
        return C;
    }

    public static double cosine(Map<Integer,Double> A, Map<Integer,Double> B) {
        double numerator = dotProduct(A, B);
        double denominator = Math.sqrt(squaredNorm(A)) * Math.sqrt(squaredNorm(B));
        return numerator/denominator;
    }

    public static TIntDoubleHashMap average(Iterator<TIntDoubleHashMap> concepts) {
        final TIntDoubleHashMap aggregated = new TIntDoubleHashMap();
        int n = 0;
        while(concepts.hasNext()) {
            TIntDoubleHashMap concept = concepts.next();
            concept.forEachEntry(new TIntDoubleProcedure() {
                public boolean execute(int key, double value) {
                    aggregated.put( key, aggregated.get(key) + value);
                    return true;
                }
            });
            n++;
        }

        // Normalize for N
        final int N = n;
        aggregated.transformValues(new TDoubleFunction() {
            public double execute(double v) {
                return v/N;
            }
        });
        aggregated.compact();
        return aggregated;
    }

    /**
     * Entropy of concept vector, each concept is assigned a chance; the more equal the chances the
     * lower the entropy (less chaos).
     * @param concept Concept vector (which will be altered by this function!)
     */
    public static double entropy(TIntDoubleHashMap concept) {
        return entropy(new TDoubleArrayList(concept.values()));
    }

    public static double entropy(double[] vector) {
        return entropy(new TDoubleArrayList(vector));
    }


        public static double entropy(TDoubleArrayList concept) {
        // The sum needs to be one
        final double sum = concept.sum();
        concept.transformValues(new TDoubleFunction() {
            public double execute(double v) {
                return v / sum;
            }
        });

        // Transform to vector
        MatrixVector p = new MatrixVector(concept.size());
        double[] values = concept.toArray();
        for(int i=0; i<values.length; i++)
            p.set(i, values[i]);

        // Get entropy
        return getEntropyA(p);
    }

    public static double sum(double[] values) {
        double sum = 0;
        for(double v : values)
            sum += v;
        return sum;
    }
}
