package ix.complexity.stanford;

import no.uib.cipr.matrix.Matrix;
import org.apache.commons.lang3.ArrayUtils;
import scala.Tuple3;

import java.util.Arrays;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 2/18/12
 * Time: 8:23 PM
 * To change this template use File | Settings | File Templates.
 */
public class ReferenceFeatures {
    public static Tuple3<Double[], Double[], Double[]> matrixToFeatures(Matrix simOriginal, int maxK, boolean weigh, boolean truncate, boolean fill, boolean normalize) {
        Double[] locals = new Double[maxK];
        Double[] globals = new Double[maxK];
        Matrix sim = simOriginal.copy();

        int size = Math.min(sim.numColumns(), sim.numRows());
        if(sim.numColumns()==0) return null;

        //Truncate
        if(truncate)
            for(int m=0; m<size; m++)
                for(int n=0; n<m; n++)
                    if(sim.get(m,n)>1)
                        sim.set(m,n,1);

        //Weigh
        if(weigh)
            for(int m=0; m<size; m++)
                for(int n=0; n<m; n++)
                    if(sim.get(m,n)>0) //no need to weigh ones that are 0
                        sim.set(m,n,sim.get(m,n)/(m-n));

        //k-steps
        double[] steps;
        steps = new double[Math.min(maxK, size)+1];
        for(int m=0; m<size; m++)
            for(int n=Math.max(0,m-steps.length+1); n<m; n++)
                steps[m-n]+=sim.get(m,n);
        //System.out.println(Arrays.toString(steps));

        //Normalize to average
        if(normalize)
            for(int i=1; i<steps.length; i++)
                steps[i] = steps[i] / Math.max(size-i, 1);
        // the max-term is to prevent division by 0. When maxK > size, the denumerator will be 0 or less
        // cause of multiple comparisons, normalize to single comparisons again

        // total sum at one, two, three, ... maxK steps
        // eg 1 1 1.25 1.25 1.5
        // meaning the final one (maxK) is the one with the most "far" view
        // fill boolean replaces nulls with preceding sum
        double ksum;
        for(int i=1; i<=maxK; i++) {
            ksum = 0;
            for(int j=1;j<=Math.min(i,steps.length-1);j++)
                ksum += steps[j];
            if(i<steps.length || fill)
                locals[i-1] = ksum;
        }

        //Avg
        for(int k=1; k<=Math.min(size,maxK); k++) { // k = 1..5           1..4 -> i=0..3
            double sum = 0;
            for(int m=0; m<k; m++) { // m = 0..k                          0..3
                for(int n=0; n<m; n++)  // n = 0..m                       0..2 
                    sum+=sim.get(m,n);
            }
            globals[k-1] = sum;
        }

        double[] stepsExt = Arrays.copyOf(steps, maxK+1); //this copies the contents of steps array to a longer,
        //System.out.println(Arrays.toString(stepsExt));
        // new array, and puts 0d for extra elements the original array didn't have
        // but the weird thing is, the length seems one less than the Double[maxK] initialization...
        Double[] stepsObj = ArrayUtils.toObject(stepsExt); //this turns into an object array
        return new Tuple3<Double[], Double[], Double[]>(stepsObj, locals, globals);
    }
}
