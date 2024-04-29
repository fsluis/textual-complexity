package ix.complexity.lucene3.esa;

import gnu.trove.function.TDoubleFunction;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.procedure.TIntDoubleProcedure;
import org.apache.commons.collections15.buffer.CircularFifoBuffer;

import java.util.Arrays;
import java.util.Iterator;

/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 1/22/14
 * Time: 2:51 PM
 * To change this template use File | Settings | File Templates.
 */
public class WindowedCentroid implements Iterator<TIntDoubleHashMap> {
    private int length;
    private CircularFifoBuffer<TIntDoubleHashMap> buffer;
    // The data in TIDHM wasn't consistent! (Really, wtf??)
    final TIntDoubleHashMap aggregated = new TIntDoubleHashMap();
    //private final HashMap<Integer,Double> aggregated = new HashMap<Integer,Double>();
    private Iterator<TIntDoubleHashMap> concepts;
    private TIntDoubleHashMap window = null; // cache, next centroid vector up
    private boolean average = false;
    //private int i = 0, j = 0;

    public WindowedCentroid(Iterator<TIntDoubleHashMap> concepts, int length, boolean average) {
        this.length = length;
        this.concepts = concepts;
        this.average = average;
        buffer = new CircularFifoBuffer<TIntDoubleHashMap>(length);
    }

    public boolean hasNext() {
        if (window==null)
            window = window();
        return (window!=null);
    }

    public TIntDoubleHashMap next() {
        if (window==null)
            window = window();
        TIntDoubleHashMap res = window;
        window = null;
        return res;
    }

    public void remove() {
        throw new UnsupportedOperationException();
    }

    private TIntDoubleHashMap window() {
        while(concepts.hasNext()) {
            //System.out.println("Concept: "+(i++));
            // Remove last value from aggregated vector, only if the buffer is already filled (i.e., all #window items
            // have been added to the buffer)
            if (buffer.isFull())
                buffer.get().forEachEntry(new TIntDoubleProcedure() {
                    public boolean execute(int i, double v) {
                        double adjust = -1 * v; // / length;
                        double agg = aggregated.adjustOrPutValue(i,adjust,0);
                        if(agg==0) aggregated.remove(i); // this is a bit of a problem, as floating-point numbers are rather unprecise/unstable? -> seems to go well!
                        //if(i==6291) System.out.println("Removing last concept from aggregated: i="+i+", v="+v+", v+="+adjust+", agg="+aggregated.get(i));
                        //if(aggregated.get(i)<=0)
                        //    aggregated.remove(i); << had to remove this cause glove vectors can be < 0
                        return true;
                    }
                });

            // Add latest concept to aggregated vector
            TIntDoubleHashMap concept = concepts.next();
            concept.forEachEntry(new TIntDoubleProcedure() {
                public boolean execute(int i, double v) {
                    double adjust = v; ///length;
                    aggregated.adjustOrPutValue(i,adjust,adjust);
                    //if(i==6291) System.out.println("Adding next concept to aggregated: i="+i+", v="+v+", v+="+adjust+", agg="+aggregated.get(i));
                    return true;
                }
            });

            buffer.add(concept);
            //System.out.println("current agg sum: "+ Arrays.stream(aggregated.values()).sum());

            // Submit the aggregated vector
            if(buffer.isFull()) {
                //System.out.println("Window concept: size="+aggregated.size()+", mass="+EsaUtil.sum(aggregated.values())+", agg(6291)="+aggregated.get(6291) );
                TIntDoubleHashMap result = new TIntDoubleHashMap(aggregated);
                if(average)
                    result.transformValues(new TDoubleFunction() {
                        public double execute(double value) {
                            return value / length;
                        }
                    });
                //System.out.println("Result: "+(j++));
                return result;
            }
        }
        return null;
    }
}
