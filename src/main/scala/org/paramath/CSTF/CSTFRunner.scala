package org.paramath.CSTF


import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.paramath.CSTF.utils.CSTFUtils

object CSTFRunner {

  var helptxt =
    """
      |Usage:
      | {input file} {cstf type}
      |
      | {String} {1|2}
      |
      | 1 => Old CSTF
      | 2 => CSTF Graph
    """.stripMargin

  def main (args: Array[String]): Unit = {
    println(util.Properties.versionString)
//    println(helptxt)

    val sparkS = new SparkConf()
        .setMaster("local[*]")
      .setAppName("CSTF_UnitTest")
    val rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)
    val sc = new SparkContext(sparkS)

    val inputFile = args(0)
    val outputFile = "CSTF_Output"
    val runType: Int = args(1).toInt
    val data :RDD[String] = sc.textFile(inputFile)
    val TensorRdd:RDD[Vector] = CSTFUtils.FileToTensor(data)


    def num_iter:  Int    = 10
    def Rank:      Int    = 100
    def tolerance: Double = 1E-12
    var rt: Double = 0.0
    var func = CSTFTree
    if (runType == 1){
      rt = CSTFTree.CP_ALS(num_iter, TensorRdd, Rank, tolerance, sc, outputFile)
      println("V1")
    } else if (runType == 2){
      rt = CSTFTreeV2.CP_ALS(num_iter, TensorRdd, Rank, tolerance, sc, outputFile)
      println("V2")
    }
    println(s"Running time: $rt")

  }

}
