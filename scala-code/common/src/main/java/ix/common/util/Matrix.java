package ix.common.util;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.io.MatrixSize;
import no.uib.cipr.matrix.io.MatrixVectorReader;
import no.uib.cipr.matrix.io.MatrixVectorWriter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 13-jan-2010
 * Time: 15:01:09
 * To change this template use File | Settings | File Templates.
 */
public class Matrix extends DenseMatrix {
    public Matrix(MatrixVectorReader matrixVectorReader) throws IOException {
        super(matrixVectorReader);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public Matrix(int i, int i1) {
        super(i, i1);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public Matrix(no.uib.cipr.matrix.Matrix matrix) {
        super(matrix);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public Matrix(no.uib.cipr.matrix.Matrix matrix, boolean b) {
        super(matrix, b);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public Matrix(Vector vector, boolean b) {
        super(vector, b);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public Matrix(Vector vector) {
        super(vector);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public Matrix(Vector[] vectors) {
        super(vectors);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public Matrix(double[][] doubles) {
        super(doubles);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public MatrixVector getRow(int j) {
        MatrixVector row = new MatrixVector(numColumns());
        for(int i=0; i<numColumns();i++) {
            row.set(i, get(j,i));
        }
        return row;
    }

    public void setRow(int j, Vector row) {
        for(int i=0; i<numColumns();i++) {
            set(j, i, row.get(i));
        }
    }

    public Matrix getRows(List<Integer> indices) {
        Matrix rows = new Matrix(indices.size(), numColumns());
        for ( int j=0; j<indices.size(); j++) {
            rows.setRow(j, getRow(indices.get(j)));
        }
        return rows;
    }

    public MatrixVector getColumn(int i) {
        MatrixVector column = new MatrixVector(numRows());
        for(int j=0; j<numRows();j++) {
            column.set(j, get(j,i));
        }
        return column;
    }

    public void setColumn(int i, Vector column) {
        for(int j=0; j<numRows();j++) {
            set(j, i, column.get(j));
        }
    }

    public Matrix removeColumn(int column) {
        Matrix copy = new Matrix(this.numRows(), this.numColumns()-1);
        int n=0;
        for(int i=0; i<numColumns(); i++)
            if(i!=column)
                copy.setColumn(n++, getColumn(i));
        return copy;
    }

    public void write(String file) throws IOException {
        MatrixVectorWriter writer = new MatrixVectorWriter(new BufferedWriter(new FileWriter(file)));
        writer.printMatrixSize(new MatrixSize(numRows(), numColumns(), numColumns() * numRows()));
        writer.printArray(getData());
        writer.close();
    }
}
