package org.paramath.CSTF

import java.io.File

import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils._
import org.paramath.structures.{IRowMatrix, TensorTree3}

import scala.util.control.Breaks

object CSTFTreeV2 {

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
    var cftotalTime:Double = 0
    val loop = new Breaks

    val Tree_CBA0 = TensorTree(TensorData, 0)
    val Tree_CBA = new TensorTree3(TensorData,0, sc)
    val Tree_CAB = new TensorTree3(TensorData,1, sc)
    val Tree_ABC = new TensorTree3(TensorData,2, sc)
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

    def Update_NFM(TreeTensor: TensorTree3,
                   mi: Array[IRowMatrix],
                   Size:Long,
                   N:Int): IRowMatrix =
    {
      var M: IRowMatrix = this.UpdateFM(TreeTensor,mi,Size,Rank,sc)
      Lambda= UpdateLambda(M,N)
      NormalizeMatrix(M,Lambda)
    }

    val time_s:Double=System.nanoTime()
    loop.breakable
    {
      for (i <- 0 until IterNum)
      {
        val mr = new Array[IRowMatrix](2)
        mr(0) = MB
        mr(1) = MC
        tick = System.currentTimeMillis()
        MA = Update_NFM(Tree_CBA,mr,MA.nRows(),i)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MA")
        tick = tock
        mr(0) = MC
        mr(1) = MA
        MB = Update_NFM(Tree_CAB,mr,MB.nRows(),i)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MB")
        tick = tock
        mr(0) = MA
        mr(1) = MB
        MC = Update_NFM(Tree_ABC,mr,MC.nRows(),i)
        tock = System.currentTimeMillis()
        printTime(tick, tock, "Update NFM, MC")

        pre_fit = fit

        tick = System.currentTimeMillis()
//        fit = 0
        var cftime = System.currentTimeMillis()
        fit = ComputeFit (
          Tree_CBA0,
          TensorData,
          Lambda,
          MA,
          MB,
          MC,
          MA.computeGramian(),
          MB.computeGramian(),
          MC.computeGramian()
        )
        var cftime2 = System.currentTimeMillis()
        cftotalTime += (cftime2 - cftime)
        tock = System.currentTimeMillis()
        printTime(tick, tock, s"Compute fit $i")
        val_fit = abs(fit - pre_fit)
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

  def ComputeFit(TreeTensor: RDD[(Vector, List[Vector])],
                 TensorData: RDD[Vector],
                 L: BDV[Double],
                 A: IRowMatrix,
                 B: IRowMatrix,
                 C: IRowMatrix,
                 ATA: BDM[Double],
                 BTB: BDM[Double],
                 CTC: BDM[Double]) = {
    val tmp: BDM[Double] = (L * L.t) :* ATA :* BTB :* CTC
    val normXest = abs(sum(tmp))
    val norm = TensorData.map(x => x.apply(3) * x.apply(3)).reduce(_ + _)

    var product = 0.0
    val Result = TreeTensor
      .map(x => (x._1(0).toLong, x))
      // .join(B.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .join(B.rows)
      .mapValues(x => (x._1._1(1).toLong, x)).values
      //      .join(C.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .join(C.rows)
      .mapValues(x => (x._1._1, x._1._2 :* x._2)).values
      .flatMap(x => x._1._2.map(v => (v(0).toLong, x._2 :*= v.apply(1))))
      //      .join(A.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .join(A.rows)
      .mapValues(v => v._1 :* v._2)
      .values
      .reduce(_ + _)

    product = product + Result.t * L
    val residue = sqrt(normXest + norm - 2 * product)
    val Fit = 1.0 - residue / sqrt(norm)


    Fit
  }

  def UpdateFM(TensorData: TensorTree3,
               mi: Array[IRowMatrix],
               SizeOfMatrix: Long,
               rank: Int,
               sc:SparkContext
              ): IRowMatrix =
  {
    TensorData.computeM1(mi,SizeOfMatrix,rank)
      .multiply(ComputeM2(mi(0),mi(1)))
  }
}
