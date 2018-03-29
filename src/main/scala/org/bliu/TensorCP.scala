package org.bliu

/**
  * Created by cqwcy201101 on 12/6/16
  */

import org.apache.log4j.{Level, Logger}
import breeze.linalg.{DenseVector => BDV}
import breeze.numerics.abs
import org.apache.spark.mllib.linalg.distributed.{IndexedRowMatrix, IndexedRow}
import org.apache.spark.{SparkConf, SparkContext}
import org.paramath.CSTF.utils.CSTFUtils

import scala.util.control.Breaks

object TensorCP {

  def main(args: Array[String]): Unit = {
    val inputFile = args(0)
    val output = "CSTF_COO_Output"

    val time_s:Double=System.nanoTime()

    val rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("TensorCP").setMaster("local[*]")
    rl.setLevel(Level.ERROR)
    val sc = new SparkContext(conf)
    rl.setLevel(Level.ERROR)
    val inputData = sc.textFile(inputFile)

    val TensorData = CSTFUtils.FileToTensor(sc.textFile(inputFile)).cache()

    val SizeVector = CloudCP.RDD_VtoRowMatrix(TensorData)
      .computeColumnSummaryStatistics().max

    val SizeA:Long = SizeVector.apply(0).toLong+1
    val SizeB:Long = SizeVector.apply(1).toLong+1
    val SizeC:Long = SizeVector.apply(2).toLong+1
    val Dim_1:Int = 0
    val Dim_2:Int = 1
    val Dim_3:Int = 2

    val iterat = args(1).toInt
    val R:Int = args(2).toInt
    val tolerance = 1E-10


    var MA:IndexedRowMatrix = CloudCP.InitialIndexedRowMatrix(SizeA,R,sc)
    var MB:IndexedRowMatrix = CloudCP.InitialIndexedRowMatrix(SizeB,R,sc)
    var MC:IndexedRowMatrix = CloudCP.InitialIndexedRowMatrix(SizeC,R,sc)

    var N:Int = 0


    var ATA = CloudCP.Compute_MTM_RowMatrix(MA)
    var BTB = CloudCP.Compute_MTM_RowMatrix(MB)
    var CTC = CloudCP.Compute_MTM_RowMatrix(MC)

    var Lambda:BDV[Double] = BDV.zeros(R)
    var fit:Double = 0.0
    var prev_fit:Double = 0.0
    var val_fit:Double = 0.0


    val loop = new Breaks

    loop.breakable{

      for (i <- 0 until  iterat)
      {
        var tick = System.currentTimeMillis()
        val cpalstick = System.currentTimeMillis()
        MA = CloudCP.UpdateMatrix(TensorData,MB,MC,Dim_1,SizeA,R,sc)
        Lambda = CloudCP.UpdateLambda(MA,i)
        MA = CloudCP.NormalizeMatrix(MA,Lambda)
        ATA = CloudCP.Compute_MTM_RowMatrix(MA)
        var tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute MA $i")

        tick = tock
        MB = CloudCP.UpdateMatrix(TensorData,MC,MA,Dim_2,SizeB,R,sc)
        Lambda =CloudCP.UpdateLambda(MB,i)
        MB = CloudCP.NormalizeMatrix(MB,Lambda)
        BTB = CloudCP.Compute_MTM_RowMatrix(MB)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute MB $i")

        tick = tock
        MC= CloudCP.UpdateMatrix(TensorData,MA,MB,Dim_3,SizeC,R,sc)
        Lambda = CloudCP.UpdateLambda(MC,i)
        MC = CloudCP.NormalizeMatrix(MC,Lambda)
        CTC= CloudCP.Compute_MTM_RowMatrix(MC)
        tock = System.currentTimeMillis()

        CSTFUtils.printTime(tick, tock, s"Compute MC $i")
        val cpalstock = System.currentTimeMillis()
        CSTFUtils.printTime(cpalstick, cpalstock, s"CP_ALS $i")
        prev_fit = fit
        tick = System.currentTimeMillis()
        fit = CloudCP.ComputeFit(TensorData,Lambda,MA,MB,MC,ATA,BTB,CTC)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute Fit $i")
        val_fit = abs(fit - prev_fit)
        val ttime = ((cpalstock-cpalstick)/1000) + ((tock-tick)/1000)
        println()
        println(s"Total CP_ALS $i $ttime")
        println(s"Fit Value $i : $val_fit")


        N = N +1

        if (val_fit < tolerance)
          loop.break
      }


    }

    //---------------
    val time_e:Double=System.nanoTime()

    //---------------


    /*
    MA.rows.collect().foreach(println)
    MB.rows.collect().foreach(println)
    MC.rows.collect().foreach(println)
    */

    println(val_fit)
    println(N)
    println(Lambda)

    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")




  }

}

