package ix.complexity.lucene3.esa;

import gnu.trove.map.hash.TIntDoubleHashMap;

public class CosineKernel extends Kernel {
    public double compare(TIntDoubleHashMap A, TIntDoubleHashMap B) {
        return similarity(A,B);
    }

    public static double similarity(TIntDoubleHashMap A, TIntDoubleHashMap B) {
        double numerator = EsaUtil.dotProduct(A, B);
        double denominator = Math.sqrt(EsaUtil.squaredNorm(A)) * Math.sqrt(EsaUtil.squaredNorm(B));
        return numerator / denominator;
    }
}
