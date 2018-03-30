package org.bliu

/**
  * Generalized COO Tensor Implementation for 3+ order tensors.
  */

import org.apache.log4j.{Level, Logger}
import breeze.linalg.{pinv, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.{abs, sqrt}
import org.bliu.CloudCP.{BDVtoVector, Compute_MTM_RowMatrix}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.paramath.CSTF.utils.CSTFUtils

import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks

object TensorCPGeneralized {

  def main(args: Array[String]): Unit = {
    val inputFile = args(0)
    val maxIterations = args(1).toInt
    val rank:Int = args(2).toInt
    val tolerance = 1E-10


    val output = "CSTF_COO_Output"

    val rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("TensorCP")
    rl.setLevel(Level.ERROR)
    val sc = new SparkContext(conf)
    rl.setLevel(Level.ERROR)
    val inputData = sc.textFile(inputFile)

    val tensorData = CSTFUtils.FileToTensor(sc.textFile(inputFile)).cache()

    CP_ALS(tensorData, maxIterations, rank, tolerance, sc)


  }

  /**
    * Perform the CP_ALS decomposition
    * @return The decomposition of the tensor.
    */
  def CP_ALS(tensorData: RDD[Vector],
             maxIterations: Int,
             rank: Int,
             tolerance: Double,
             sc: SparkContext): Double = {

    val tensor = tensorData.cache()
    val sizeVector = CloudCP.RDD_VtoRowMatrix(tensor)
      .computeColumnSummaryStatistics().max
    val dims = tensorData.first().size - 1

    val time_s: Long = System.nanoTime()
    val maxDimSizes: ListBuffer[Long] = ListBuffer[Long]()
    for(i <- 0 until sizeVector.size - 1) {
      maxDimSizes += (sizeVector(i) + 1).toLong
    }

    val matrices = new Array[IndexedRowMatrix](dims)
    for(i <- 0 until matrices.length) {
      matrices(i) = CloudCP.InitialIndexedRowMatrix(maxDimSizes(i), rank, sc)
    }

    var N:Int = 0

    var lambda:BDV[Double] = BDV.zeros(rank)
    var fit:Double = 0.0
    var prev_fit:Double = 0.0
    var val_fit:Double = 0.0


    val loop = new Breaks

    loop.breakable{

      for (i <- 0 until maxIterations)
      {
        var tick = System.currentTimeMillis()
        val cpalstick = System.currentTimeMillis()
        var tock = System.currentTimeMillis()

        // Compute the updates for all matrices
        for(j <- 0 until matrices.length) {
          tick = System.currentTimeMillis()

          val mats = dropAndRotate(matrices, j)
          matrices(j) = mttkrpProduct(tensor,
            mats,
            rank,
            j,
            matrices(j).numRows(),
            sc)
          lambda = CloudCP.UpdateLambda(matrices(j), i)
          matrices(j) = CloudCP.NormalizeMatrix(matrices(j), lambda)

          tock = System.currentTimeMillis()
          CSTFUtils.printTime(tick, tock, s"Compute M$j $i")


        }

        val cpalstock = System.currentTimeMillis()
        CSTFUtils.printTime(cpalstick, cpalstock, s"CP_ALS $i")
        prev_fit = fit
        tick = System.currentTimeMillis()
//        fit = computeFit(tensor, lambda, matrices)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute Fit $i")
        val_fit = abs(fit - prev_fit)
        val ttime: Double = ((cpalstock-cpalstick)/1000.0) + ((tock-tick)/1000.0)
        println()
        println(s"Total CP_ALS $i $ttime")
        println(s"Fit Value $i : $val_fit")


        N = N +1

//        if (val_fit < tolerance)
//          loop.break
      }


    }

    //---------------
    val time_e:Double=System.nanoTime()

    //---------------

    println(val_fit)
    println(N)
    println(lambda)

    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")
    0.0
  }

  def computeFit(tensor: RDD[Vector],
                 lambda: BDV[Double],
                 mats: Array[IndexedRowMatrix]): Double = {
    var tmp:BDM[Double] = (lambda * lambda.t)
    for (i <- 0 until mats.length) {
      tmp :*= Compute_MTM_RowMatrix(mats(i))
    }
    val normXest = abs(sum(tmp))
    val dims = tensor.first().size - 1
    val norm = tensor.map(x => x(dims)*x(dims)).reduce(_+_)

    var product = 0.0

    var joinedVecs: RDD[(Long, (Vector, BDV[Double]))] = tensor.map(v => (v(0).toLong, v))
      .join(mats(0).rows.map(x=> (x.index, vecToBDV(x.vector))))
      .map(f => (f._2._1(1).toLong, (f._2)))
    for(i <- 1 until mats.length) {
      joinedVecs = joinedVecs
        .join(mats(i).rows.map(f => (f.index, vecToBDV(f.vector))))
        .map({
          case(ind: Long, ((v: Vector, vOld: BDV[Double]), vNew: BDV[Double])) => {
            (v(i+1).toLong, (v, vOld :* vNew))
          }})
    }
    val t1: RDD[(BDV[Double])] = joinedVecs.map({
      case(i: Long, (v: Vector, vOld: BDV[Double])) => {
        vOld :* v(v.size-1)
      }})
    val result = t1.reduce(_ + _)
    product += (result.t * lambda)
    val residue = sqrt(normXest + norm - (2 * product))
    val fit = 1.0 - (residue/sqrt(norm))
    fit
  }

  /**
    * Drops the i-th index from the array and rotates the array so that i+1
    * in the original array is now the 0th index and i-1 is the last index.
    * @param v - The original array
    * @param i - Index to drop and rotate at
    * @return
    */
  def dropAndRotate(v: Array[IndexedRowMatrix], i: Int): Array[IndexedRowMatrix] = {
    val sz = v.length
    val (first, last) = v.splitAt(i % sz)
    (last ++ first).slice(1, v.length)
  }

  def mttkrp(tensor: RDD[Vector],
                mats: Array[IndexedRowMatrix],
                dim: Int,
                numRows: Long,
                rank: Int,
                sc: SparkContext): IndexedRowMatrix = {
    // Assume matrices in array are passed in correct order
    val InitialM1: IndexedRowMatrix = CloudCP.GenM1(numRows,rank,sc)

    // Indexes from 0 to size - 2 ::: v(size-1) is the tensor value, v(dim) is the dim we don't join on
    // i.e. 3rd order tensor will have vector size 4
    // Thus our indexes should range from 0 to 2 during mttkrp
    val totalDims: Int = tensor.first().size - 1 // when using mod (%) gives values between [0, 2]
    var tmp: RDD[(Long, (Vector, BDV[Double]))] = tensor
      .map(v => (v( (dim+1) % totalDims  ).toLong, v))
      .join(mats(0).rows.map(f => (f.index, vecToBDV(f.vector)))) // join the 0th matrix
      .map({
        case(ind: Long, (v: Vector, data: BDV[Double])) => (v((dim + 2) % totalDims).toLong, (v, data))
      })

    for(i <- 1 until mats.length - 1) {
      val t1 = tmp.join(mats(i).rows.map(f => (f.index, vecToBDV(f.vector))))
      tmp = t1.map({ case(ind: Long, ((vec: Vector, vOld: BDV[Double]), vNew: BDV[Double])) => {
        (vec( ((dim+1)+ i + 1) % totalDims).toLong, (vec, vOld :* vNew))
      }})
    }
    val result: RDD[(Long, BDV[Double])] = tmp.join(mats(mats.length-1).rows.map(f => (f.index, vecToBDV(f.vector))))
      .map({ case(ind: Long, ((v: Vector, vOld: BDV[Double]), vNew: BDV[Double])) => {
          (v(dim).toLong, vOld :* vNew :* v(totalDims))
        }
      }).reduceByKey(_ + _)

    val tempM1: RDD[(Long, Vector)] = InitialM1.rows.map(
          x => (x.index, vecToBDV(x.vector) ))
          .cogroup(result)
          .mapValues{x =>
            if (x._2.isEmpty) {
              BDVtoVector(x._1.head)
            }  else {
              BDVtoVector(x._2.head)
            }
          }

    val ResultM1:IndexedRowMatrix = new IndexedRowMatrix(
      tempM1.map(x => {
        IndexedRow(x._1,Vectors.dense(x._2.toArray))
      }))
    ResultM1

  }

  def vecToBDV(v1: Vector): BDV[Double] = {
    new BDV(v1.toArray)
  }

  def computeM2(mats: Array[IndexedRowMatrix]): BDM[Double] = {
    var x1 = Compute_MTM_RowMatrix(mats(0))
    for(i <- 1 until mats.length) {
      x1 = x1 :* Compute_MTM_RowMatrix(mats(i))
    }
    pinv(x1)
  }

  def mttkrpProduct(tensor: RDD[Vector],
                    mats: Array[IndexedRowMatrix],
                    rank: Int,
                    dim: Int,
                    nRows: Long,
                    sc:SparkContext): IndexedRowMatrix =
  {
    mttkrp(tensor, mats, dim, nRows, rank, sc).multiply(CloudCP.BDMtoMatrix(computeM2(mats)))
  }
}


