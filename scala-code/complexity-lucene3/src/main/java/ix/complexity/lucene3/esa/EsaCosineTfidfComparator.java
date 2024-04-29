package ix.complexity.lucene3.esa;

import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.procedure.TDoubleProcedure;
import ix.common.util.LimitedCapacityQueue;
import org.apache.lucene3.index.IndexReader;
import org.apache.lucene3.index.Term;
import org.apache.lucene3.index.TermDocs;

import java.io.IOException;

/**
 * Implements the Explicit Semantic Analysis (ESA) relatedness measure by Gabrilovich & Markovitch (2007)
 * gm(w1,w2) = cosine (cv1, cv2),
 *   where cv is the so called concept vector of a word.
 *   A concept vector is derived from the occurences of the term in Wikipedia concepts (=articles).
 *   If a term occurs in an Wikipedia article, the tfidf weight of the term in this article is the entry for that article in the concept vector.
 *   
 * The method can also be applied to whole texts. The concept vectors for a text are itemwise averaged (thus forming a centroid vector).
 * The centroid vectors are used as aggregated concept vectors.  
 * 
 * Depending on the parameters that were used to create the underlying Lucene index, 
 *   it will be necessary to preprocess terms before querying relatedness with the comparator.
 *   E.g. if the Lucene index was stemmed, query terms should also be stemmed. 
 * 
 * @author zesch
 *
 */
public class EsaCosineTfidfComparator extends EsaComparator implements RelatednessComparable {

    public EsaCosineTfidfComparator(String index, double weightingThreshold) throws EsaException {
        super(index, weightingThreshold);
    }

    /**
     *
     *
     * @param token The search term.
     * @return A mapping from documents that contain a given term to the term's tfidf-weight in that document.
     * @throws EsaException
     */
    public TIntDoubleHashMap getConceptMap(String token) throws EsaException {
        Term term = new Term(fieldName, token);


        // get enumerations of the documents with term1/term2
        TermDocs termDocs = null;
        try {
            termDocs = reader.termDocs(term);
        } catch (IOException e) {
            throw new EsaException(e);
        }


        // get the document frequencies of term 
        int df = 0;
        try {
            df = reader.docFreq(term);
        } catch (IOException e) {
            throw new EsaException(e);
        }

        // if df is zero, then the term is not in the index
        if (df == 0) {
            return new TIntDoubleHashMap(0);
        }

        TIntDoubleHashMap conceptMap = new TIntDoubleHashMap(df);

        // calculate inverse document frequency (idf)
        int numDocs = reader.numDocs();
        double idf = Math.log((double) numDocs / df);

        double tfidf;
        try {
            while(termDocs.next()) {
                int documentID = termDocs.doc();
                int wikiID = Integer.parseInt(reader.document(documentID).getField("id").stringValue());
                int tf = termDocs.freq();
                double logTermFreq;
                if (tf > 0) {
                    logTermFreq = 1 + Math.log( tf );
                    tfidf = logTermFreq * idf;
                    conceptMap.put(wikiID, tfidf);
                }
                //else {
                //    logTermFreq = 0;
                //}

            }
        } catch (IOException e) {
            throw new EsaException(e);
        }
        
        try {
            termDocs.close();
        } catch (IOException e) {
            throw new EsaException(e);
        }

//        System.out.println("unnorm");
//        for (int id : conceptMap.keySet()) {
//            System.out.println(id + " - " + conceptMap.get(id));
//        }
//        System.out.println();
        TDoubleArrayList tfidfs = new TDoubleArrayList(conceptMap.values());
        tfidfs.sort();
        final LimitedCapacityQueue<Double> queue = new LimitedCapacityQueue<Double>(100);
        final double c = .05*tfidfs.get(0);
        tfidfs.forEachDescending(new TDoubleProcedure() {
            public boolean execute(double b) {
                Double a = queue.append(b);
                //if(a!=null && (b-a)<c)
                //    return false;   //This is theory B: it's probably what gabrilovic proposes...
                if(a!=null && b/a>.95)
                    return false;   //This is (my) theory A: it's less harsh on the minimum slope
                return true;
            }
        });
        //double threshold = queue.peekFirst();
        double threshold = queue.peekLast();
        applyThreshold(conceptMap, threshold);
        
        normalizeConceptMap(conceptMap);    // normalize the vector
        applyThreshold(conceptMap, this.weightingThreshold);         // apply the weighting threshold

//        System.out.println("norm");
//        for (int id : conceptMap.keySet()) {
//            System.out.println(id + " - " + conceptMap.get(id));
//        }
//        System.out.println();

        return conceptMap;
    }

   public String getFieldName() {
        return fieldName;
    }

    public IndexReader getReader() {
        return reader;
    }
}