package org.paramath.CSTF

/**
  * Created by cqwcy201101 on 4/28/17.
  */

import org.paramath.structures.IRowMatrix

import org.paramath.CSTF.utils.CSTFUtils._
import org.paramath.BLAS
import breeze.linalg.{DenseMatrix  => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD

import scala.util.control.Breaks
import org.apache.commons.io.FileUtils
import java.io.File

object CSTFTree {


  /**
    * Does the compute
    * @param IterNum
    * @param TensorData
    * @param Rank
    * @param Tolerance
    * @param sc
    * @param outputPath
    * @return The execution time
    */
  def CP_ALS(IterNum: Int,
             //TreeTensor:RDD[(Vector,List[Vector])],
             TensorData:RDD[Vector],
             Rank:Int,
             Tolerance:Double,
             sc:SparkContext,
             outputPath:String): Double =
  {

    var tick, tock = System.currentTimeMillis();
    val loop = new Breaks

    val Tree_CBA = TensorTree(TensorData,0).cache()
    val Tree_CAB = TensorTree(TensorData,1).cache()
    val Tree_ABC = TensorTree(TensorData,2).cache()
    tock = System.currentTimeMillis()
    printTime(tick, tock, "Caching")
    tick = tock

    val SizeVector = RDD_DVtoRowMatrix(TensorData).computeColumnSummaryStatistics().max
    tock = System.currentTimeMillis()
    printTime(tick, tock, "SizeVector")
    tick = tock
    val OrderSize = List(SizeVector(0).toLong+1,SizeVector(1).toLong+1,SizeVector(2).toLong+1)
    tock = System.currentTimeMillis()
    printTime(tick, tock, "OrderSize")

    tick = tock
    var MA = Randomized_IRM(OrderSize(0),2,sc)
    var MB = Randomized_IRM(OrderSize(1),2,sc)
    var MC = Randomized_IRM(OrderSize(2),2,sc)
    var Lambda: BDV[Double] = BDV.zeros(Rank)
    tock = System.currentTimeMillis()
    printTime(tick, tock, "Matrix Generation")

    var fit = 0.0
    var pre_fit = 0.0
    var val_fit = 0.0
    var N:Int = 1

    def Update_NFM(TreeTensor:RDD[(Vector,List[Vector])] ,
                   m1: IRowMatrix,
                   m2:IRowMatrix,
                   Size:Long,
                   N:Int): IRowMatrix =
    {
      var M: IRowMatrix = UpdateFM(TreeTensor,m1,m2,Size,Rank,sc)
      Lambda= UpdateLambda(M,N)
      M = NormalizeMatrix(M,Lambda)

      M
    }

    val time_s:Double=System.nanoTime()
    loop.breakable
    {
      for (i <- 0 until IterNum)
      {
        tick = System.currentTimeMillis()
        MA = Update_NFM(Tree_CBA,MB,MC,MA.nRows(),i)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MA")
        tick = tock
        MB = Update_NFM(Tree_CAB,MC,MA,MB.nRows(),i)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MB")
        tick = tock
        MC = Update_NFM(Tree_ABC,MA,MB,MC.nRows(),i)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MC")

        pre_fit = fit

        tick = System.currentTimeMillis()
        fit = ComputeFit (
          Tree_CBA,
          TensorData,
          Lambda,
          MA,
          MB,
          MC,
          MA.computeGramian(),
          MB.computeGramian(),
          MC.computeGramian()
        )
        tock = System.currentTimeMillis()
        printTime(tick, tock, s"Compute fit $i")
        val_fit = abs(fit - pre_fit)
        N = N+1

        if (val_fit<Tolerance)
          loop.break
      }
    }
    val time_e:Double=System.nanoTime()
    val runtime = (time_e - time_s)/1000000000 + "s"

    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")
    println("fit = " + val_fit)
    println("Iteration times = " + N)

    val RDDTIME = sc.parallelize(List(runtime))
    FileUtils.deleteDirectory(new File(outputPath))
    RDDTIME.distinct().repartition(1).saveAsTextFile(outputPath)



    (time_e - time_s)/1000000000
  }




}
