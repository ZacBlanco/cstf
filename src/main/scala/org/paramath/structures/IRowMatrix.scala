package org.paramath.structures


import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.rdd.RDD

class IRowMatrix(var rows: RDD[(Long, BDV[Double])],
                 var numRows: Long = -1,
                 var numCols: Long = -1) {



}
