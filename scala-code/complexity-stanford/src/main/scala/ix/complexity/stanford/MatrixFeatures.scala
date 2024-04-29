package ix.complexity.stanford

import no.uib.cipr.matrix.{Matrices, Matrix}

import scala.collection.immutable.Range

object MatrixFeatures {
  def matrixToWindows(sim: Matrix, windowSize: Int): IndexedSeq[Matrix] = {
    val size = math.min(sim.numColumns, sim.numRows)
    for (i <- 1 to size) yield {
      val rows = Range(Math.max(i - windowSize, 0), i).toArray
      val cols = Range(Math.max(i - windowSize, 0), i).toArray
      Matrices.getSubMatrix(sim, rows, cols)
    }
  }
}
