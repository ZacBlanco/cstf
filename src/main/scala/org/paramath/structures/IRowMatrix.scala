package org.paramath.structures


import org.paramath.BLAS
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateOnlineSummarizer, MultivariateStatisticalSummary}

class IRowMatrix(var rows: RDD[(Long, BDV[Double])],
                 var numRows: Long = -1,
                 var numCols: Int = -1) {

  def this(rows: RDD[(Long, BDV[Double])]) {
    this(rows, -1, -1)
    rows.cache()
  }


  def nCols(): Int = {
    if (numCols == -1){
      computeCols()
    }
    numCols
  }

  def nRows(): Long = {
    if (numRows == -1) {
      computeRows()
    }
    numRows
  }

  def computeRows(): Unit = {
    if (numRows == -1) {
      numRows = rows.count()
    }
  }

  def computeCols(): Unit = {
    if (numCols == -1) {
      numCols = rows.first()._2.length
    }
  }
  def computeGramian(): BDM[Double] = {
    val n: Int = nCols()
    checkNumColumns(n)
    // Computes n*(n+1)/2, avoiding overflow in the multiplication.
    // This succeeds when n <= 65535, which is checked above
    val nt: Int = if (n % 2 == 0) ((n / 2) * (n + 1)) else (n * ((n + 1) / 2))

    // Compute the upper triangular part of the gram matrix.
    val GU = rows.treeAggregate(new BDV[Double](nt))(
      seqOp = (U, v) => {
        BLAS.spr(1.0, Vectors.dense(v._2.toArray), U.data)
        U
      }, combOp = (U1, U2) => U1 += U2)

    triuToFull(n, GU.data)
  }

  private def triuToFull(n: Int, U: Array[Double]): BDM[Double] = {
    val G = new BDM[Double](n, n)

    var row = 0
    var col = 0
    var idx = 0
    var value = 0.0
    while (col < n) {
      row = 0
      while (row < col) {
        value = U(idx)
        G(row, col) = value
        G(col, row) = value
        idx += 1
        row += 1
      }
      G(col, col) = U(idx)
      idx += 1
      col +=1
    }

    G
  }

  private def checkNumColumns(cols: Long): Unit = {
    if (cols > 65535) {
      throw new IllegalArgumentException(s"Argument with more than 65535 cols: $cols")
    }
//    if (cols > 10000) {
//      val memMB = (cols.toLong * cols) / 125000
//      logWarning(s"$cols columns will require at least $memMB megabytes of memory!")
//    }
  }

  /**
    * Computes column-wise summary statistics.
    */
  def computeColumnSummaryStatistics(): MultivariateStatisticalSummary = {
    val summary = rows.map(f => Vectors.dense(f._2.toArray)).
      treeAggregate(new MultivariateOnlineSummarizer)(
//      (aggregator, data) => aggregator.add(data),
      (aggregator, data) => aggregator.add(data),
      (aggregator1, aggregator2) => aggregator1.merge(aggregator2))
    summary
  }

  /**
    * Multiply this matrix by a local matrix on the right.
    *
    * @param B a local matrix whose number of rows must match the number of columns of this matrix
    * @return a [[org.paramath.structures.IRowMatrix]] representing the product,
    *         which preserves partitioning
    */
  def multiply(B: BDM[Double]): IRowMatrix = {
    val n: Int = nCols()
    val k: Int = B.cols
    require(n == B.rows, s"Dimension mismatch: $n vs ${B.rows}")

    require(B.isInstanceOf[BDM[Double]],
      s"Only support dense matrix at this time but found ${B.getClass.getName}.")

    val Bb = rows.context.broadcast(B.toDenseVector.toArray)
    val AB = rows.mapPartitions { iter =>
      val Bi = Bb.value
      iter.map { row =>
        val v = BDV.zeros[Double](k)
        var i = 0
        while (i < k) {
          v(i) = row._2.dot(new BDV(Bi, i * n, 1, n))
          i += 1
        }
        (row._1, v)
      }
    }

    new IRowMatrix(AB, numRows, B.cols)
  }
}
