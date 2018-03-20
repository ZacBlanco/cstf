package org.paramath.structures

import breeze.linalg.rank
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils.{ComputeM2, mttkrp}

object TT3Util {

  /**
    * Calculates the next index for the tensor.
    * @param dims number of dimensions in the tensor
    * @param v current tensor index
    * @return The index in the vector of the next tensor value
    */
  def nextTensorIndex(dims: Int, v: Int): Int = {
    //    var x = v
    //    if (x + 1 == mode) {
    //      x = (x + 2) % dims
    //    } else {
    //      x = (x+1) % dims
    //    }
    //    x
    (v + 1) % dims
  }

  /**
    * Gets the gets the ID of the tree this tensor is a part of
    * @param dims number of dimensions in the tensor
    * @param mode - mode of the tensor
    * @return the ID of the tree
    */
  def getTreeID(dims: Int, mode: Int): Int = {
    nextTensorIndex(dims, mode)
  }

  /**
    * Root index is always the mode index + 1 (or 0 if wraps around)
    * @param dims number of dimensions in the tensor
    * @param mode Mode of the tensor
    * @param v The value
    * @return
    */
  def isRootIndex(dims: Int, mode: Int, v: Int): Boolean = {
    (mode + 1) % dims == v
  }

  def getRootIndex(dims: Int, mode: Int): Int = {
    (mode + 1) % dims
  }

  /**
    * Will always be the last index if the index + 1 (with mod wrap around) is the mode
    * @param dims
    * @param mode
    * @param v
    * @return
    */
  def isLastIndex(dims: Int, mode: Int, v: Int): Boolean = {
    (v + 1) % dims == mode
  }

  /**
    * List of VecInfo reduce for final stage
    * @param a
    * @param b
    * @return
    */
  def vr(a: VecInfo, b: VecInfo): VecInfo = {
    VecInfo(0, a.vec + b.vec)
  }

  def mttkrpProduct(TensorData: TensorTree3,
                    mi: Array[IRowMatrix],
                    SizeOfMatrix: Long,
                    rank: Int,
                    sc:SparkContext
                   ): IRowMatrix =
  {
    val M1 = TensorData.mttkrp(mi,SizeOfMatrix,rank)
    val M2 = ComputeM2(mi(0),mi(1))
    M1.multiply(M2)
  }

}
