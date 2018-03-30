package org.paramath.CSTF

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.bliu.TensorCPGeneralized
import org.paramath.CSTF.utils.CSTFUtils
import org.paramath.SparkTestSuite

class CSTFTreeGeneralizedSuite extends SparkTestSuite{

  //    val testFile = "random1.txt"
//  val testFile = "random2.txt"
//  val testFile = "nell2-small.tns"
//  val testFile = "enron-med.tns"
  val testFile = "vast-med.tns"
//  val testFile = "vast-small.tns"
  val numTrials = 1

  test("Tree Generalized") {
    CPALS_test(CSTFTreeGeneralized.CP_ALS)

  }

  test("NEW C00 OPTIMIZED") {
    val Data:RDD[String] = sc.textFile(testFile)
    val tensor:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def maxIter:  Int    = 15
    def rank:      Int    = 2
    def tolerance: Double = 1E-10
    COOGeneralized.CP_ALS(tensor, maxIter, rank, tolerance, sc)
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
