package org.paramath.CSTF

import java.io.File

import breeze.linalg.{rank, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils._
import org.paramath.structures.{IRowMatrix, TT3Util, TensorTree3}

import scala.util.control.Breaks

object CSTFTreeV2 {

  /**
    * Does the compute
    * @param IterNum
    * @param TensorData
    * @param rank
    * @param Tolerance
    * @param sc
    * @param outputPath
    * @return The execution time
    */
  def CP_ALS(IterNum: Int,
             TensorData:RDD[Vector],
             rank:Int,
             Tolerance:Double,
             sc:SparkContext,
             outputPath:String): Double =
  {

    var tick, tock = System.currentTimeMillis();
    var cftotalTime:Double = 0
    val loop = new Breaks

    val Tree_CBA0 = TensorTree(TensorData, 0)
    val Tree_JKI = new TensorTree3(TensorData,0, sc)
    val Tree_KIJ = new TensorTree3(TensorData,1, sc)
    val Tree_IJK = new TensorTree3(TensorData,2, sc)
    tock = System.currentTimeMillis()
    printTime(tick, tock, "Caching")
    tick = tock

    val sizeVector = RDD_DVtoRowMatrix(TensorData).computeColumnSummaryStatistics().max
    tock = System.currentTimeMillis()
    printTime(tick, tock, "SizeVector")
    tick = tock
    val OrderSize = List(sizeVector(0).toLong+1,sizeVector(1).toLong+1,sizeVector(2).toLong+1)
    tock = System.currentTimeMillis()
    printTime(tick, tock, "OrderSize")

    tick = tock
    var MI = Randomized_IRM(OrderSize(0),rank,sc)
    var MJ = Randomized_IRM(OrderSize(1),rank,sc)
    var MK = Randomized_IRM(OrderSize(2),rank,sc)
    var lambda: BDV[Double] = BDV.zeros(rank)
    tock = System.currentTimeMillis()
    printTime(tick, tock, "Matrix Generation")

    var fit = 0.0
    var pre_fit = 0.0
    var val_fit = 0.0
    var N:Int = 1

    val time_s:Double=System.nanoTime()
    loop.breakable
    {
      for (i <- 0 until IterNum)
      {
        val mr = new Array[IRowMatrix](2)
        mr(0) = MJ
        mr(1) = MK
        tick = System.currentTimeMillis()
        MI = TT3Util.mttkrpProduct(Tree_JKI, mr, MI.nRows(), rank, sc)
        lambda = updateLambda(MI, i)
        MI = normalizeMatrix(MI, lambda)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MA")



        tick = tock
        mr(0) = MK
        mr(1) = MI
        MJ = TT3Util.mttkrpProduct(Tree_KIJ, mr, MJ.nRows(), rank, sc)
        lambda = updateLambda(MJ, i)
        MJ = normalizeMatrix(MJ, lambda)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MB")




        tick = tock
        mr(0) = MI
        mr(1) = MJ
        MK = TT3Util.mttkrpProduct(Tree_IJK, mr, MK.nRows(), rank, sc)
        lambda = updateLambda(MK, i)
        MK = normalizeMatrix(MK, lambda)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MC")

        pre_fit = fit

        tick = System.currentTimeMillis()
//        fit = 0
        var cftime = System.currentTimeMillis()
        fit = computeFit2(
          Tree_CBA0,
          TensorData,
          lambda,
          MI,
          MJ,
          MK,
          MI.computeGramian(),
          MJ.computeGramian(),
          MK.computeGramian()
        )
        var cftime2 = System.currentTimeMillis()
        cftotalTime += (cftime2 - cftime)
        tock = System.currentTimeMillis()
        printTime(tick, tock, s"Compute fit $i")
        val_fit = abs(fit - pre_fit)
        println(s"Fit is $val_fit")
        N = N+1

        if (val_fit<Tolerance)
          loop.break
      }
    }
    val time_e:Double=System.nanoTime()
    cftotalTime /= 1000
    val rtime = (((time_e - time_s)/1000000000) - cftotalTime)
    val runtime = rtime + "s"

    println(s"Running time is: $rtime")
    println(s"Compute fit time: $cftotalTime")
    println("fit = " + val_fit)
    println("Iteration times = " + N)
    println()

    val RDDTIME = sc.parallelize(List(runtime))
    FileUtils.deleteDirectory(new File(outputPath))
    RDDTIME.distinct().repartition(1).saveAsTextFile(outputPath)

    rtime
  }
//
//  def ComputeFit(TreeTensor: RDD[(Vector, List[Vector])],
//                 TensorData: RDD[Vector],
//                 L: BDV[Double],
//                 A: IRowMatrix,
//                 B: IRowMatrix,
//                 C: IRowMatrix,
//                 ATA: BDM[Double],
//                 BTB: BDM[Double],
//                 CTC: BDM[Double]) = {
//    val tmp: BDM[Double] = (L * L.t) :* ATA :* BTB :* CTC
//    val normXest = abs(sum(tmp))
//    val norm = TensorData.map(x => x.apply(3) * x.apply(3)).reduce(_ + _)
//
//    var product = 0.0
//    val Result = TreeTensor
//      .map(x => (x._1(0).toLong, x))
//      // .join(B.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
//      .join(B.rows)
//      .mapValues(x => (x._1._1(1).toLong, x)).values
//      //      .join(C.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
//      .join(C.rows)
//      .mapValues(x => (x._1._1, x._1._2 :* x._2)).values
//      .flatMap(x => x._1._2.map(v => (v(0).toLong, x._2 :*= v.apply(1))))
//      //      .join(A.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
//      .join(A.rows)
//      .mapValues(v => v._1 :* v._2)
//      .values
//      .reduce(_ + _)
//
//    product = product + Result.t * L
//    val residue = sqrt(normXest + norm - 2 * product)
//    val Fit = 1.0 - residue / sqrt(norm)
//
//
//    Fit
//  }
}
