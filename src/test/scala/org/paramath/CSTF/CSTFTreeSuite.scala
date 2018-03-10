package org.paramath.CSTF

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils
import org.paramath.SparkTestSuite

class CSTFTreeSuite extends SparkTestSuite {

  var testFile = "nell2-small.tns"
  var numTrials = 1

  test("random1.txt v1 Test") {


  val inputFile = testFile
    val outputFile = "CSTF_Output"

    val Data:RDD[String] = sc.textFile(inputFile)
    val TensorRdd:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def num_iter:  Int    = 10
    def Rank:      Int    = 100
    def tolerance: Double = 1E-12


    var min: Double = Double.MaxValue
    var max: Double = Double.MinValue
    var avg: Double = 0.0
    for (i <- 0 until numTrials) {
      var rt = org.paramath.CSTF.CSTFTree.CP_ALS(num_iter, TensorRdd, Rank, tolerance, sc, outputFile)
      if (rt < min)
        min = rt
      if (rt > max)
        max = rt
      avg += rt
    }

    avg /= 11
    println(s"The min running time: $min")
    println(s"The max running time: $max")
    println(s"The avg running time: $avg")

  }

  test("random1.txt v2 Test") {


    val inputFile = testFile
    val outputFile = "CSTF_Output"

    val Data:RDD[String] = sc.textFile(inputFile)
    val TensorRdd:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def num_iter:  Int    = 10
    def Rank:      Int    = 100
    def tolerance: Double = 1E-12


    var min: Double = Double.MaxValue
    var max: Double = Double.MinValue
    var avg: Double = 0.0
    for (i <- 0 until numTrials){
      var rt = CSTFTreeV2.CP_ALS(num_iter, TensorRdd, Rank, tolerance, sc, outputFile)
      if (rt < min)
        min = rt
      if (rt > max)
        max = rt
      avg += rt
    }

    avg /= 11
    println(s"The min running time: $min")
    println(s"The max running time: $max")
    println(s"The avg running time: $avg")

  }

}
