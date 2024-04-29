package ix.complexity.lucene3.esa;

import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.map.hash.TIntDoubleHashMap;
import org.apache.lucene3.index.IndexReader;
import org.apache.lucene3.search.IndexSearcher;
import org.apache.lucene3.store.Directory;
import org.apache.lucene3.store.FSDirectory;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Abstract base class for Explicit Semantic Analysis (ESA) based relatedness measures following Gabrilovich & Markovitch (2007).
 *
 * @author zesch
 *
 */
public abstract class EsaComparator implements RelatednessComparable {
    
//    private final static Logger logger = Logger.getLogger(EsaComparator.class);

    protected IndexSearcher searcher;
    protected IndexReader reader;
    protected String fieldName;
    protected double weightingThreshold;

    public EsaComparator(String index, double weightingThreshold) throws EsaException {
        
        this.weightingThreshold = weightingThreshold;
        this.fieldName = "content"; //this is the name of the index field!

        try {
            // get the location of the index files
            // second parameter false means not to erase existing files
            System.out.println("Opening dir+ "+index);
            Directory dir = FSDirectory.open(new File(index));
            System.out.println("Opened dir+ "+index);
            //if (IndexReader.isLocked(dir)) {
            //    IndexReader.unlock(dir);
            //}
            searcher = new IndexSearcher(dir, true);
            reader = searcher.getIndexReader();
        } catch (IOException e) {
            throw new EsaException(e);
        }

    }

    /* (non-Javadoc)
     * @see org.tud.ukp.wikipedia.api.relatedness.RelatednessComparable#getRelatedness(java.lang.String, java.lang.String)
     */
    public double getRelatedness(String token1, String token2) throws EsaException {

        // Do not represent vectors as real vectors => they will be almost always sparse
        // We can compare entries by the concept's document id.  
        // Convert token to lower case, because the index is also lower case.
        TIntDoubleHashMap conceptMap1 = getConceptMap(token1.toLowerCase());
        TIntDoubleHashMap conceptMap2 = getConceptMap(token2.toLowerCase());
        
        return computeEsaRelatedness(conceptMap1, conceptMap2);
    }

    /**
     * Get ESA relatedness of a whole texts.
     * @param tokenList1 The first text as a list of tokens.
     * @param tokenList2 The second text as a list of tokens.
     * @return The relatedness between the two texts.
     * @throws EsaException
     */
    public double getRelatedness(List<String> tokenList1, List<String> tokenList2) throws EsaException {
       TIntDoubleHashMap averageConceptMap1 = getAverageConceptMap(tokenList1);
       TIntDoubleHashMap averageConceptMap2 = getAverageConceptMap(tokenList2);

//       logger.info(CommonUtilities.getMapContents(averageConceptMap1));
//       logger.info(CommonUtilities.getMapContents(averageConceptMap2));

       return computeEsaRelatedness(averageConceptMap1, averageConceptMap2);
    }


    /**
     *
     *
     * @param token The search term.
     * @return A mapping from a concept to a value representing the tokens importance for that document.
     * The value may be binary, tfidf weight, etc.
     * @throws EsaException
     */
    protected abstract TIntDoubleHashMap getConceptMap(String token) throws EsaException;


    /**
     * @param tokenList The search terms.
     * @return A mapping from documents that contain a given term to the term's tfidf-weight in that document.
     * @throws EsaException
     */
    private TIntDoubleHashMap getAverageConceptMap(List<String> tokenList) throws EsaException {
        TIntDoubleHashMap averageConceptMap = new TIntDoubleHashMap(tokenList.size()*10);
        
        // sum all concept weights
        for (String token : tokenList) {
            // convert token to lower case, because the index is also lower case
            TIntDoubleHashMap conceptWeightMap = getConceptMap(token.toLowerCase());
            
            if (conceptWeightMap == null) {
                throw new EsaException("ConceptMap is null - should not happen!");
            }
            
            for (int key : conceptWeightMap.keys()) {
                if (averageConceptMap.containsKey(key)) {
                    averageConceptMap.put(key, averageConceptMap.get(key) + conceptWeightMap.get(key));
                }
                else {
                    averageConceptMap.put(key, conceptWeightMap.get(key));
                }
            }
        }

        // divide sums by number of tokens
        for (int key : averageConceptMap.keys()) {
            averageConceptMap.put(key, averageConceptMap.get(key) / tokenList.size());
        }
        
        return averageConceptMap;
    }    
    
    /**
     * Compute the vector product of two sparse vectors represented as maps. The key indicates the position in the vector.
     * @param scv1 The first sparse vector represented as a map. 
     * @param scv2 The second sparse vector represented as a map.
     * @return
     */
    private double sparseVectorProduct(TIntDoubleHashMap scv1, TIntDoubleHashMap scv2) {
        double vectorProduct = 0.0;
        
        for (int key : scv1.keys()) {
            if (scv2.containsKey(key)) {
                vectorProduct += scv1.get(key) * scv2.get(key);
            }
                
        }
        
        return vectorProduct;
    }
    
    /**
     * @param scv The sparse vector.
     * @return The normalization value for a sparse vector represented as a map.
     */
    private double getL2Norm(TIntDoubleHashMap scv) {
        
        double sumOfSquares = 0.0;
        for (int key : scv.keys()) {
            sumOfSquares += scv.get(key) * scv.get(key);
        }
        
        return Math.sqrt(sumOfSquares);
    }

    /**
     * Computes the ESA relatedness by computing the cosine between two concept vectors. 
     * @param conceptMap1 The first concept vector.
     * @param conceptMap2 The second concept vector.
     * @return The ESA relatedness of the two concept vectors.
     */
    private double computeEsaRelatedness(TIntDoubleHashMap conceptMap1, TIntDoubleHashMap conceptMap2) {
        // return relatedness of -1, if one of the terms is not in the index (then one of the conceptWeightMaps is null)
        if (conceptMap1 == null || conceptMap2 == null) {
            return -1;
        }

        // If one of the concept maps is empty, the cosine is zero.
        // An empty concept map leads to a 0 normValue that is caught as an error afterwards.
        // Thus, we check that here.
        if (conceptMap1.size() == 0 || conceptMap2.size() == 0) {
            return 0.0;
        }
        
        // cosine(a,b) =  axb / sqrt(a^2 * b^2)
        double normValue1 = getL2Norm(conceptMap1);
        double normValue2 = getL2Norm(conceptMap2);
        
        if (normValue1 <= 0.0 || normValue2 <= 0.0) {
            return -1;
        }
        
        double cosine = sparseVectorProduct(conceptMap1, conceptMap2) / (normValue1 * normValue2);

        return cosine;
    }
    
    /**
     * Normalizes a concept vector.
     * @param conceptMap The concept vector to normalize.
     */
    protected void normalizeConceptMap(TIntDoubleHashMap conceptMap) {

        double l2Norm = getL2Norm(conceptMap);
        
        if (l2Norm == 0.0) {
            return;
        }
        
        for (int key : conceptMap.keys()) {
           conceptMap.put(key, conceptMap.get(key) / l2Norm);
        }        
    }
    

    /**
     * Apply the weighting threshold to the concept map.
     * This removes all elements in the map with values lower or equal to the threshold.
     * @param conceptMap The concept vector.
     */
    protected void applyThreshold(TIntDoubleHashMap conceptMap, double weightingThreshold) {
        if(weightingThreshold <= 0)
            return;
        TIntDoubleIterator iter = conceptMap.iterator();
        while (iter.hasNext()) {
            iter.advance();
           if (iter.value() <= weightingThreshold) {
               iter.remove();
           }
        }
        conceptMap.compact();
    }
}