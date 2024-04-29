package ix.complexity.features.common;

import no.uib.cipr.matrix.Matrix;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 2/18/12
 * Time: 8:23 PM
 * To change this template use File | Settings | File Templates.
 */
public class ReferenceFeatures {
    private int maxK = 5;
    private boolean weigh = true, truncate = false;

    /**
     * Calculates cohesion over steps maxK.
     * @param sim similarity matrix, holding a similarity value for each couple of units. Only the square part of the
     *            matrix is used: size=min(columns,rows). Any extra row/col is ignored.
     * @return cohesion[0] is global cohesion (whole matrix), cohesion[1..maxK] the avg cohesion over 1..maxK units
     */
    public double[] cohesion(Matrix sim) {
        double[] cohesion = new double[maxK+1];
        int size = Math.min(sim.numColumns(), sim.numRows());
        if(sim.numColumns()==0) return cohesion;

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
                    sim.set(m,n,sim.get(m,n)/(m-n));

        //k-steps
        double[] steps;
        steps = new double[Math.min(maxK, size)+1];
        for(int m=0; m<size; m++)
            for(int n=Math.max(0,m-steps.length+1); n<m; n++)
                steps[m-n]+=sim.get(m,n);

        //Normalize to average
        //for(int i=1; i<steps.length; i++)
        //    steps[i] = steps[i]/(size-i);
        //what's this supposed to do???

        //Sum k-steps to step j
        double ksum;
        for(int i=1; i<steps.length; i++) {
            ksum = 0;
            for(int j=1;j<=i;j++)
                ksum += steps[j];
            if(ksum>=0)
                cohesion[i]=ksum;
        }

        //Avg
        double sum = 0;
        for(int m=0; m<size; m++)
            for(int n=0; n<m; n++)
                sum+=sim.get(m,n);
        double avg = sum / ((size*size-1)/2);
        cohesion[0] = avg;

        return cohesion;
    }

    public void setMaxK(int maxK) {
        this.maxK = maxK;
    }

    public void setWeigh(boolean weigh) {
        this.weigh = weigh;
    }

    public void setTruncate(boolean truncate) {
        this.truncate = truncate;
    }
}
