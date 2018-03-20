package org.paramath.CSTF

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils
import org.paramath.SparkTestSuite
import breeze.linalg.{DenseVector => BDV}

class CSTFTreeSuite extends SparkTestSuite {

  var testFile = "nell2-small.tns"
//  var testFile = "random2.txt"
  var numTrials = 1

  test("v1 Test") {
    CPALS_test(CSTFTree.CP_ALS)
  }

  test("v2 Test") {
//    val l1 = IndexedSeq.fill(5)(0)
//    val l2 = IndexedSeq.fill(5)(0)
//    l1.updated(0, 6)
//    println(l1 == l2)
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
