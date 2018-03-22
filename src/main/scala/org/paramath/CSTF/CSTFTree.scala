package org.paramath.CSTF

/**
  * Created by cqwcy201101 on 4/28/17.
  */

import java.io.File

import breeze.numerics.{abs, log}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils._
import org.paramath.structures.TensorTree

import scala.util.control.Breaks

object CSTFTree {


  /**
    * Does the compute
    *
    * @param IterNum
    * @param TensorData
    * @param rank
    * @param Tolerance
    * @param sc
    * @param outputPath
    * @return The execution time
    */
  def CP_ALS(IterNum: Int,
             TensorData: RDD[Vector],
             rank: Int,
             Tolerance: Double,
             sc: SparkContext,
             outputPath: String): Double = {

    var tick, tock = System.currentTimeMillis();
    var cftotalTime: Double = 0
    val loop = new Breaks

    val Tree_BCA = new TensorTree(sc, TensorData, 0)
    val Tree_CAB = new TensorTree(sc, TensorData, 1)
    val Tree_ABC = new TensorTree(sc, TensorData, 2)
    tick = tock

    val SizeVector = RDD_DVtoRowMatrix(TensorData).computeColumnSummaryStatistics().max
    tock = System.currentTimeMillis()
    printTime(tick, tock, "SizeVector")
    val OrderSize = List(SizeVector(0).toLong + 1, SizeVector(1).toLong + 1, SizeVector(2).toLong + 1)

    tock = System.currentTimeMillis()
    tick = tock
    var MA = Randomized_IRM(OrderSize(0), rank, sc)
    var MB = Randomized_IRM(OrderSize(1), rank, sc)
    var MC = Randomized_IRM(OrderSize(2), rank, sc)
    var lambda: BDV[Double] = BDV.zeros(rank)
    tock = System.currentTimeMillis()
    printTime(tick, tock, "Matrix Generation")

    var fit = 0.0
    var pre_fit = 0.0
    var val_fit = 0.0
    var N: Int = 0

    val time_s: Double = System.nanoTime()
    loop.breakable {
      for (i <- 0 until IterNum) {

        val cpalstick: Long = System.currentTimeMillis()
//        println("Computing A")
        tick = System.currentTimeMillis()
        MA = mttkrpProduct(Tree_BCA, MB, MC, MA.nRows(), rank, sc)
        lambda = updateLambda(MA, i)
        MA = normalizeMatrix(MA, lambda)
        tock = System.currentTimeMillis()
        printTime(tick, tock, s"Compute MA $i")

//        println("Computing B")
        tick = tock
        MB = mttkrpProduct(Tree_CAB, MC, MA, MB.nRows(), rank, sc)
        lambda = updateLambda(MB, i)
        MB = normalizeMatrix(MB, lambda)
        tock = System.currentTimeMillis()
        printTime(tick, tock, s"Compute MB $i")


//        println("Computing C")
        tick = tock
        MC = mttkrpProduct(Tree_ABC, MA, MB, MC.nRows(), rank, sc)
        lambda = updateLambda(MC, i)
        MC = normalizeMatrix(MC, lambda)
        tock = System.currentTimeMillis()
        printTime(tick, tock, s"Compute MC $i")
        val cpalstock: Long = System.currentTimeMillis()
        printTime(cpalstick, cpalstock, s"CP_ALS $i")

        pre_fit = fit
        tick = System.currentTimeMillis()
        val cftick = System.currentTimeMillis()
        fit = computeFit(
          Tree_BCA,
          TensorData,
          lambda,
          MA,
          MB,
          MC,
          MA.computeGramian(),
          MB.computeGramian(),
          MC.computeGramian()
        )
        val cftock = System.currentTimeMillis()
        cftotalTime += (cftock - cftick).toDouble
        tock = System.currentTimeMillis()
        printTime(cftick, cftock, s"Compute fit $i")

        val_fit = abs(fit - pre_fit)
        println(s"Fit $i is $val_fit")
        val totalIterTime = ((cpalstock-cpalstick) / 1000) + ((cftock - cftick)/1000)
        println(s"Total CP_ALS time $i $totalIterTime")
        N = N + 1

        if (val_fit < Tolerance)
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


}
