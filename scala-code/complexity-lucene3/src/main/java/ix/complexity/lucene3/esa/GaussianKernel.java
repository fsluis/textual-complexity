package ix.complexity.lucene3.esa;

import gnu.trove.map.hash.TIntDoubleHashMap;

public class GaussianKernel extends Kernel {
        private double sigma = 1, denominator, normalizer;

        public GaussianKernel(double sigma) {
            this.sigma = sigma;
            denominator = 2 * Math.pow(sigma, 2);
            normalizer = 1 / Math.sqrt(2 * Math.PI * sigma);
        }

        public double compare(TIntDoubleHashMap A, TIntDoubleHashMap B) {
            double numerator = -1 * EsaUtil.squaredNorm(EsaUtil.min(A, B));
            return normalizer * Math.exp(numerator / denominator);
            // Cf http://en.wikipedia.org/wiki/Kernel_(statistics) and shawe-taylor 2007
        }

        public double getSigma() {
            return sigma;
        }
    }
