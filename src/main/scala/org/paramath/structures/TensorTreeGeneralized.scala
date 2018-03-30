package org.paramath.structures

import breeze.linalg.{max, DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.HashPartitioner

class TensorTreeGeneralized(sc: SparkContext) {
  var dims: Int = 0
  var dim: Int = 0
  var trees: Array[RDD[Vector]] = new Array(dims)
  var tree: RDD[(Vector, List[Vector])] = sc.emptyRDD
  var tensor1: RDD[(Long, (Vector, List[Vector]))] = sc.emptyRDD

  def this(sc: SparkContext, tensor: RDD[Vector], dim: Int = 0) {
    this(sc)
    this.dim = dim
    this.dims = tensor.first().size - 1

    // This arrays holds the value of each index rotated for the corresponding dimension
    // that this tree corresponds to.
    val inds: Array[Int] = new Array[Int](this.dims-1)
    for (i <- 1 until this.dims) { // Until is DIFFERENT than to in scala loops
      inds(i-1) = (dim + i) % this.dims
    }

    var d: Int = this.dims
    this.tree = tensor.map(v =>
      (Vectors.dense(inds.map(i => v(i))), Vectors.dense(v(dim), v(d)))


    )
      .combineByKey(List(_),
        (c: List[Vector], v: Vector) => v :: c,
        (c1: List[Vector], c2: List[Vector]) => c1 ::: c2).cache()

    this.tensor1 = this.tree
      .map(pair => (pair._1(0).toLong, pair))
      .cache()
  }

  def mttkrp(mats: Array[IRowMatrix]): IRowMatrix = {


    var tmp: RDD[(Long, ((Vector, List[Vector]), BDV[Double]))] = this.tensor1.join(mats(0).rows)
      .map({
        case(i_old: Long, ((inds: Vector, vals: List[Vector]), vecNew: BDV[Double])) => {
          (inds(1).toLong, ((inds, vals), vecNew))
        }
      })

    for (i <- 1 until mats.length - 1){
     tmp = TensorTreeGeneralized.joinMultiplyIncrementIndex(tmp, i+1, mats(i)) // join with mat(i), then next RDD has vec(i+1) as key
    }
    val msplit: RDD[(Long, BDV[Double])] = tmp.join(mats(mats.length-1).rows).flatMap({
      case(i_old: Long, (((inds: Vector, vals: List[Vector]), vecOld: BDV[Double]), vecNew: BDV[Double])) => {
        val finalVec = vecOld :* vecNew
        vals.map(vec => (vec(0).toLong, finalVec :* vec(1)))
      }
    }).reduceByKey(_ + _)
    new IRowMatrix(msplit)

  }

}

object TensorTreeGeneralized {
  def joinMultiplyIncrementIndex(currRDD: RDD[(Long, ((Vector, List[Vector]), BDV[Double]))],
                                 index: Int,
                                 matrix: IRowMatrix): RDD[(Long, ((Vector, List[Vector]), BDV[Double]))] = {
    currRDD.join(matrix.rows).map({
      case(i_old: Long, (((inds: Vector, vals: List[Vector]), vecOld: BDV[Double]), vecNew: BDV[Double])) => {
        (inds(index).toLong, ((inds, vals), vecOld :* vecNew))
      }
    })
  }
}
