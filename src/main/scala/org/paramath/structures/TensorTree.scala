package org.paramath.structures

import breeze.linalg.{max, DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.HashPartitioner

class TensorTree(sc: SparkContext) {
  var dims: Int = 0
  var dim: Int = 0
  var trees: Array[RDD[Vector]] = new Array(dims)
  var tree: RDD[(Vector, List[Vector])] = sc.emptyRDD
  var tensor1: RDD[(Long, (Vector, List[Vector]))] = sc.emptyRDD

  def this(sc: SparkContext, tensor: RDD[Vector], dim: Int = 0) {
    this(sc)
    this.dim = dim
    this.dims = tensor.first().size - 1
    val inds: Array[Int] = new Array[Int](this.dims-1)
    for (i <- 1 until this.dims) { // Until is DIFFERENT than to
      inds(i-1) = (dim + i) % this.dims
    }

    var d: Int = this.dims
    this.tree = tensor.map(v => (Vectors.dense(inds.map(i => v(i))), Vectors.dense(v(dim), v(d))))
      .combineByKey(List(_),
        (c: List[Vector], v: Vector) => v :: c,
        (c1: List[Vector], c2: List[Vector]) => c1 ::: c2).cache()

    this.tensor1 = this.tree
      .map(pair => (pair._1(0).toLong, pair))
      .cache()
  }

  def mttkrp(m1: IRowMatrix,
             m2: IRowMatrix,
             Rank: Int): IRowMatrix = {

    val Map_m1 = m1.rows //m1.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector)))
    val Map_m2 = m2.rows //.map(idr => (idr.index.toLong, VtoBDV(idr.vector)))

    val Join_m1 = this.tensor1
      .join(Map_m1)
      .map(line => (line._2._1._1(1).toLong, (line._2._1._2, line._2._2)))

    val t1 = Join_m1.join(Map_m2)
    val t2 = t1.map(v => (v._2._1._1, v._2._1._2 :* v._2._2))
//    val t2 = t1.mapValues(v => {
//        (v._1._1, v._1._2 :* v._2)
//      }).values
    val t3 = t2.flatMap(pair => {
      pair._1.map(v => {
        var x = pair._2 :* v(1)
        (v(0).toLong, x)
      })
    })
//      .sortByKey()
    val t4 = t3.reduceByKey(_ + _)

    new IRowMatrix(t4)
  }
}
