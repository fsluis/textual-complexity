package ix.complexity.lucene3.esa;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import gnu.trove.map.hash.TIntDoubleHashMap;
import ix.common.util.Matrix;
import ix.common.util.TimeLogger;
import no.uib.cipr.matrix.LowerSymmBandMatrix;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 3/31/11
 * Time: 11:57 AM
 * To change this template use File | Settings | File Templates.
 */
public abstract class Kernel implements Serializable {
    abstract public double compare(TIntDoubleHashMap A, TIntDoubleHashMap B);

    private static final Log log = LogFactory.getLog(Kernel.class);
    private static final TimeLogger tlog = new TimeLogger(log, "kernel matrix");

    public LowerSymmBandMatrix matrixx(List<TIntDoubleHashMap> concepts, int maxK) {
        long t = tlog.start();

        LowerSymmBandMatrix matrix = new LowerSymmBandMatrix(concepts.size(), maxK);
        for (int r = 0; r < concepts.size(); r++) { //row index
            for (int c = Math.max(r-maxK, 0); c < r; c++) { //column index (should be below row index for LowerSymm matrix)
                double similarity = compare(concepts.get(r), concepts.get(c));
                matrix.set(r, c, similarity);
            }
        }
        tlog.stop(t);
        return matrix;
    }

    public Matrix matrix(Iterable<TIntDoubleHashMap> concepts) throws IOException {
        long t = tlog.start();
        HashMultiset<TIntDoubleHashMap> distinctConcepts = HashMultiset.create(concepts);
        Matrix matrix = new Matrix(distinctConcepts.elementSet().size(), distinctConcepts.elementSet().size() + 1);
        int i = 0, j = 0;
        //log.info("Creating kernel matrix of " + distinctConcepts.elementSet().size() + "^2, instead of " + concepts.size() + "^2 for key:" + collector.getHistory());
        for (Multiset.Entry<TIntDoubleHashMap> A : distinctConcepts.entrySet()) {
            for (Multiset.Entry<TIntDoubleHashMap> B : distinctConcepts.entrySet()) {
                if (j > i) {
                    double similarity = compare(A.getElement(), B.getElement());
                    if (similarity == 1)
                        System.out.println("sim=1 @ " + i + "," + j);
                    matrix.set(i, j, similarity);
                    matrix.set(j, i, similarity);
                }
                j++;
            }
            //i += A.getCount();
            matrix.set(i, j, A.getCount());
            i++;
            j = 0;
        }

        tlog.stop(t);
        return matrix;
    }
}


