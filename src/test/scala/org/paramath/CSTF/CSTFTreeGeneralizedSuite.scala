package org.paramath.CSTF

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.scheduler
import org.apache.spark.scheduler.StageInfo
import org.bliu.TensorCPGeneralized
import org.paramath.CSTF.utils.{CSTFUtils, TaskInfoRecorderListener}
import org.paramath.SparkTestSuite

import scala.collection.mutable

class CSTFTreeGeneralizedSuite extends SparkTestSuite{

      val testFile = "random1.txt"
//  val testFile = "random2.txt"
//  val testFile = "nell2-small.tns"
//  val testFile = "enron-med.tns"
//  val testFile = "vast-med.tns"
//  val testFile = "vast-small.tns"
//  val testFile = "nell2-med.tns"
  val numTrials = 1

  test("Tree Generalized") {
    CPALS_test(CSTFTreeGeneralized.CP_ALS)

  }

  test("Q SIZE") {
    val m = new mutable.Queue[Vector]()
    m.enqueue(Vectors.dense(1, 2))
    m.enqueue(Vectors.dense(1, 3))
    m.enqueue(Vectors.dense(1, 4))
    m.enqueue(Vectors.dense(1, 5))
    val est = org.apache.spark.util.SizeEstimator
    val size = est.estimate(m)

    println(s"Size of vector queue: $size")
    val vs = est.estimate(Vectors.dense(1, 2, 3, 4))
    println(s"size of single vector $vs")
    val varr = m.toArray
    varr.foreach(println(_))
    for(i <- 0 until varr.length-1) varr(i) = varr(i+1)
    varr.foreach(println(_))
    val arrsz = est.estimate(m.toArray)
    println(s"Array size $arrsz")
//    scala.Vector[Vector]().drop()
    println(size/vs)

  }

  test("NEW C00 OPTIMIZED") {
    val listener = new org.paramath.CSTF.utils.TaskInfoRecorderListener(false)
    sc.addSparkListener(listener)

    val Data:RDD[String] = sc.textFile(testFile)
    val tensor:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def maxIter:  Int    = 5
    def rank:      Int    = 2
    def tolerance: Double = 1E-10
    COOGeneralizedRowMatrix.CP_ALS(tensor, maxIter, rank, tolerance, sc)

    listener.taskMetricsData.foreach(t => {
      TaskInfoRecorderListener.taskValsString(t)
    })
    println("Shuffle read bytes: " + listener.gatherTaskVals("remoteBytesRead"))
    println("FetchWaitTime: " + listener.gatherTaskVals("fetchWaitTime"))
    println("local bytes read: " + listener.gatherTaskVals("localBytesRead"))
    println("sada: " + listener.gatherTaskVals("asdasd"))
//    TaskInfoRecorderListener.printTaskVals(listener.gatherTaskVals())
  }


  test("Even more C00 OPTIMIZED") {
    val Data:RDD[String] = sc.textFile(testFile)
    val tensor:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def maxIter:  Int    = 15
    def rank:      Int    = 2
    def tolerance: Double = 1E-10
    COOGeneralizedSingleVec.CP_ALS(tensor, maxIter, rank, tolerance, sc)
  }

  test("COO Generalized") {
    val Data:RDD[String] = sc.textFile(testFile)
    val tensor:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def maxIter:  Int    = 20
    def rank:      Int    = 2
    def tolerance: Double = 1E-10
    TensorCPGeneralized.CP_ALS(tensor, maxIter, rank, tolerance, sc)
  }


  def CPALS_test(callback: (RDD[Vector], Int, Int, Double, SparkContext, String) => Double): Unit = {
    val inputFile = testFile
    val outputFile = "CSTF_Output"

    val Data:RDD[String] = sc.textFile(inputFile)
    val TensorRdd:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def num_iter:  Int    = 10
    def rank:      Int    = 2
    def tolerance: Double = 1E-10


    var min: Double = Double.MaxValue
    var max: Double = Double.MinValue
    var avg: Double = 0.0
    for (i <- 0 until numTrials){
      var rt = callback(TensorRdd, num_iter, rank, tolerance, sc, outputFile)
      if (rt < min)
        min = rt
      if (rt > max)
        max = rt
      avg += rt
    }

    avg /= numTrials
    println(s"The min running time: $min")
    println(s"The max running time: $max")
    println(s"The avg running time: $avg")

  }
}
