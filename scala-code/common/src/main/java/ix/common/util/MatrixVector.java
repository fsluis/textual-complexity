package ix.common.util;

import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.io.MatrixVectorReader;

import java.io.IOException;
import java.util.Iterator;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 13-jan-2010
 * Time: 16:11:54
 * To change this template use File | Settings | File Templates.
 */
public class MatrixVector extends DenseVector {
    public MatrixVector(MatrixVectorReader matrixVectorReader) throws IOException {
        super(matrixVectorReader);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public MatrixVector(int i) {
        super(i);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public MatrixVector(Vector vector) {
        super(vector);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public MatrixVector(Vector vector, boolean b) {
        super(vector, b);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public MatrixVector(double[] doubles, boolean b) {
        super(doubles, b);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public MatrixVector(double[] doubles) {
        super(doubles);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public double[] toArray() {
        double[] array = new double[this.size()];
        Iterator<VectorEntry> iterator = iterator();
        while(iterator.hasNext()) {
            VectorEntry entry = iterator.next();
            array[entry.index()] = entry.get();
        }
        return array;
    }
}
