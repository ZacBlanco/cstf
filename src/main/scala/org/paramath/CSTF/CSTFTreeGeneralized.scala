package org.paramath.CSTF

/**
  * Created by cqwcy201101 on 4/28/17.
  */

import java.io.File

import breeze.numerics.{abs, log, sqrt}
import breeze.linalg.{norm, product, rank, sum, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils.{RDD_DVtoRowMatrix, printTime, Randomized_IRM, updateLambda, normalizeMatrix}
import org.paramath.structures.{IRowMatrix, TensorTree, TensorTreeGeneralized}

import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks

object CSTFTreeGeneralized {


  /**
    * Does the compute
    *
    * @param tensorData
    * @param maxIterations
    * @param rank
    * @param tolerance
    * @param sc
    * @param outputPath
    * @return The execution time
    */
  def CP_ALS(tensorData: RDD[Vector],
             maxIterations: Int,
             rank: Int,
             tolerance: Double,
             sc: SparkContext,
             outputPath: String): Double = {

    var tick, tock = System.currentTimeMillis();
    var cftotalTime: Double = 0
    val loop = new Breaks

    val dims = tensorData.first().size - 1
    val tensorTrees: Array[TensorTreeGeneralized] = new Array[TensorTreeGeneralized](dims)
    for (i <- 0 until tensorTrees.length) {
      tensorTrees(i) = new TensorTreeGeneralized(sc, tensorData, i)
    }

    tick = tock

    val dimSizes = RDD_DVtoRowMatrix(tensorData).computeColumnSummaryStatistics().max
    tock = System.currentTimeMillis()
    printTime(tick, tock, "SizeVector")
    var maxDimSizes: ListBuffer[Long] = ListBuffer[Long]()
    for (i <- 0 until dimSizes.size) {
      maxDimSizes += (dimSizes(i) + 1).toLong
    }

    tock = System.currentTimeMillis()
    tick = tock
    val matrices: Array[IRowMatrix] = new Array[IRowMatrix](dims)
    for (i <- 0 until dims) {
      matrices(i) = Randomized_IRM(maxDimSizes(i), rank, sc)
    }

    var lambda: BDV[Double] = BDV.zeros(rank)
    tock = System.currentTimeMillis()
    printTime(tick, tock, "Matrix Generation")

    var fit = 1.0
    var pre_fit = 0.0
    var val_fit = 0.0
    var N: Int = 0

    val time_s: Double = System.nanoTime()
    loop.breakable {
      for (i <- 0 until maxIterations) {

        val cpalstick: Long = System.currentTimeMillis()
        //        println("Computing A")
        for (j <- 0 until tensorTrees.length) {
          tick = System.currentTimeMillis()

          matrices(j) = mttkrpProduct(tensorTrees(j), dropAndRotate(matrices, j), rank, sc)
          lambda = updateLambda(matrices(j), i)
          matrices(j) = normalizeMatrix(matrices(j), lambda)

          tock = System.currentTimeMillis()
          printTime(tick, tock, s"Compute M$j $i")
        }


        val cpalstock: Long = System.currentTimeMillis()
        printTime(cpalstick, cpalstock, s"CP_ALS $i")

        pre_fit = fit
        tick = System.currentTimeMillis()
        val cftick = System.currentTimeMillis()
//        fit = computeFit(tensorTrees(0),
//          tensorData,
//          lambda,
//          matrices,
//          sc)
        val cftock = System.currentTimeMillis()
        cftotalTime += (cftock - cftick).toDouble
        tock = System.currentTimeMillis()
        printTime(cftick, cftock, s"Compute fit $i")

        val_fit = abs(fit - pre_fit)
        println(s"Fit $i is $val_fit")
        val totalIterTime: Double = ((cpalstock-cpalstick) / 1000.0) + ((cftock - cftick)/1000.0)
        println()
        println(s"Total CP_ALS $i $totalIterTime")
        N = N + 1

        if (val_fit < tolerance)
          loop.break
      }
    }
    val time_e: Double = System.nanoTime()
    cftotalTime /= 1000 // The number of seconds to run compute fit
    val rtime = (((time_e - time_s) / 1000000000) - cftotalTime) //
    val runtime = rtime + "s"

    println(s"Running time is: $rtime")
    println(s"Compute fit time: $cftotalTime")
    println(s"fit = $val_fit")
    println(s"Iteration times = $N")
    println()

    val RDDTIME = sc.parallelize(List(runtime))
    //    FileUtils.deleteDirectory(new File(outputPath))
    //    RDDTIME.distinct().repartition(1).saveAsTextFile(outputPath)

    rtime
  }

  /**
    * Drops the i-th index from the array and rotates the array so that i+1
    * in the original array is now the 0th index and i-1 is the last index.
    * @param v - The original array
    * @param i - Index to drop and rotate at
    * @return
    */
  def dropAndRotate(v: Array[IRowMatrix], i: Int): Array[IRowMatrix] = {
    val sz = v.length
    val (first, last) = v.splitAt(i % sz)
    (last ++ first).slice(1, v.length)
  }

  /**
    * Compute the fit after the MTTKRP operation
    * @param tree
    * @param originalTensor
    * @param lambda
    * @param mats
    * @param sc
    * @return
    */
  def computeFit(tree: TensorTreeGeneralized,
                 originalTensor: RDD[Vector],
                 lambda: BDV[Double],
                 mats: Array[IRowMatrix],
                 sc: SparkContext): Double = {
    var bdm: BDM[Double] = lambda * lambda.t
    val dims = mats.length
    val tmp = mats.map(m => m.computeGramian())
    for (i <- 0 until mats.length) {
      bdm :*= tmp(i)
    }

    val normEstimate: Double = abs(sum(bdm))
    val norm: Double = originalTensor.map(x => x(dims) * x(dims)).reduce(_ + _) //dims is the index of the tensor entry value
    var prod: Double = 0.0

    // Similar to mttkrp implementation
    val ms = dropAndRotate(mats, 0)
    var jrdd: RDD[(Long, ((Vector, List[Vector]), BDV[Double]))] = tree.tensor1.join(ms(0).rows)
      .map({
        case(i_old: Long, ((inds: Vector, vals: List[Vector]), vecNew: BDV[Double])) => {
          (inds(1).toLong, ((inds, vals), vecNew))
        }
      })

    for(i <- 1 until ms.length - 1) {
      jrdd = TensorTreeGeneralized.joinMultiplyIncrementIndex(jrdd, i+1, ms(i))
    }
    val result = jrdd.join(ms(ms.length-1).rows).flatMap({
      case(i_old: Long, (((inds: Vector, vals: List[Vector]), vecOld: BDV[Double]), vecNew: BDV[Double])) => {
        vals.map(v => (v(0).toLong, vecOld :* vecNew :* v(1)))
      }
    })
      .join(mats(0).rows).map({
        case(i: Long, (v1: BDV[Double], v2: BDV[Double])) => {
          v1 :* v2
        }
      })
      .reduce(_ + _)
    prod += result.t * lambda
    val residue: Double = sqrt(normEstimate + norm - (2 * prod))
    val fit: Double = 1.0 - (residue / sqrt(norm))
    fit




//    var product = 0.0
//    val result = tree.tree.map(x => (x._1(0).toLong, x)).join(J.rows)
//    val r1 = result.mapValues(x => (x._1._1(1).toLong, x)).values
//    val r2 = r1.join(K.rows).mapValues(x => (x._1._1, x._1._2 :* x._2)).values
//    val r3 = r2.flatMap(x => x._1._2.map(v => (v(0).toLong, x._2 :* v(1))))
//    val r4 = r3.join(I.rows).mapValues(v => v._1 :* v._2)
//
//    val res2 = r4.values.reduce(_ + _)
//
//    product = product + res2.t * L
//    val residue = sqrt(normXest + norm - 2 * product)
//    val Fit = 1.0 - residue / sqrt(norm)
//
//    Fit

  }

  /**
    * Multiply the resulting MTTKRP by the inverse of resulting gramian matrices.
    * @param tt
    * @param mats
    * @param rank
    * @param sc
    * @return
    */
  def mttkrpProduct(tt: TensorTreeGeneralized,
                    mats: Array[IRowMatrix],
                    rank: Int,
                    sc:SparkContext): IRowMatrix =
  {
    tt.mttkrp(mats).multiply(computeM2(mats))
  }

  /**
    * Find the inverse of the gramians multiplied together
    * @param mats
    * @return
    */
  def computeM2(mats: Array[IRowMatrix]): BDM[Double] = {
    var x1: BDM[Double] = mats(0).computeGramian()
    for (i <- 1 until mats.length) {
      x1 :*= mats(i).computeGramian()
    }
    var singular: Boolean = false
    try{
      x1 = breeze.linalg.inv(x1)
    } catch {
      case mse: breeze.linalg.MatrixSingularException => singular = true
    }
    if (singular) {
      x1 = breeze.linalg.pinv(x1)
    }

    x1
  }


}
