package org.paramath.CSTF

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.CSTF.utils.CSTFUtils
import org.paramath.SparkTestSuite

class CSTFTreeSuite extends SparkTestSuite {

  test("random1.txt Test") {


    val inputFile = "random1.txt"
    val outputFile = "CSTF_Output.txt"

    val Data:RDD[String] = sc.textFile(inputFile)
    val TensorRdd:RDD[Vector] = CSTFUtils.FileToTensor(Data)


    def num_iter:  Int    = 10
    def Rank:      Int    = 100
    def tolerance: Double = 1E-12

    CSTFTree.CP_ALS(num_iter, TensorRdd, Rank, tolerance, sc, outputFile)





  }

}
