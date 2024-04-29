package ix.complexity.lucene3.esa;

import gnu.trove.map.hash.TIntDoubleHashMap;
import ix.common.util.LimitedCapacityHashMap;
import ix.common.util.TimeLogger;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.Collections;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 11/20/13
 * Time: 4:21 PM
 * To change this template use File | Settings | File Templates.
 */
public class EsaModel extends EsaCosineTfidfComparator {
    public final static Pattern PATTERN = Pattern.compile("[a-zA-Z']+");
    public static final int CACHE_SIZE = 5000; //most articles are < 5000 tokens / concepts
    public static final double WEIGHTING_THRESHOLD = 0; //most articles are < 5000 tokens / concepts

    private static final Log log = LogFactory.getLog(EsaModel.class);
    private static final TimeLogger tlog = new TimeLogger(log, "esa", 100000);

    private Map<String, TIntDoubleHashMap> cache;

    public EsaModel(String index, double weightingThreshold, int cacheSize) throws EsaException {
        super(index, weightingThreshold);
        cache = Collections.synchronizedMap(new LimitedCapacityHashMap<String, TIntDoubleHashMap>(cacheSize));
    }

    /**
     * Get esa vector
     * @param token The search term.
     * @return Esa vector, or null if token doesn't match PATTERN
     * @throws EsaException
     */
    @Override
    public TIntDoubleHashMap getConceptMap(String token) throws EsaException {
        if ((PATTERN.matcher(token).matches())) {
            long t = tlog.start();
            TIntDoubleHashMap concept;
            if (cache.containsKey(token))
                concept = cache.get(token);
            else {
                concept = super.getConceptMap(token);
                cache.put(token, concept);   //cache it
            }
            tlog.stop(t);
            if (!concept.isEmpty())
                return concept;
        }
        return null;
    }
}
