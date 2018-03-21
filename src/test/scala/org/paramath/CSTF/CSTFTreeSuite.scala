package org.paramath.CSTF

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils
import org.paramath.SparkTestSuite
import breeze.linalg.{DenseVector => BDV}

class CSTFTreeSuite extends SparkTestSuite {

  var testFile = "nell2-med.tns"
//  var testFile = "random2.txt"
  var numTrials = 1

  test("v1 Test") {
    CPALS_test(CSTFTree.CP_ALS)
  }

  test("v2 Test") {
    CPALS_test(CSTFTreeV2.CP_ALS)
  }

  def CPALS_test(callback: (Int, RDD[Vector], Int, Double, SparkContext, String) => Double): Unit = {
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
      var rt = callback(num_iter, TensorRdd, rank, tolerance, sc, outputFile)
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
