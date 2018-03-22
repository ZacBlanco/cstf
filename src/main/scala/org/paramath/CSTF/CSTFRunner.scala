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
      | {inputFile} {NumIterations} {rank}
      |
    """.stripMargin

  def main (args: Array[String]): Unit = {
    println(util.Properties.versionString)

    val sparkS: SparkConf = new SparkConf().setAppName("CSTF_TensorTree")

    val rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)
    val sc = new SparkContext()

    val inputFile = args(0)
    val outputFile = "CSTF_Output"
    val runType: Int = args(1).toInt
    val data :RDD[String] = sc.textFile(inputFile)
    val TensorRdd:RDD[Vector] = CSTFUtils.FileToTensor(data)


    def num_iter:  Int    = args(1).toInt
    def Rank:      Int    = args(2).toInt
    def tolerance: Double = 1E-12
    var rt: Double = 0.0
    println("V1")
    rt = CSTFTree.CP_ALS(num_iter, TensorRdd, Rank, tolerance, sc, outputFile)
    println(s"Running time: $rt")

  }

}
