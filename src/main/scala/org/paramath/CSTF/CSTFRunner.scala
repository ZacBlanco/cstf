package org.paramath.CSTF


import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.bliu.TensorCPGeneralized
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
    print("Using scala version: ")
    println(util.Properties.versionString)

    val inputFile: String = args(0)
    val maxIterations:  Int    = args(1).toInt
    val rank:      Int    = args(2).toInt
    var version: Int = args(3).toInt
    val tolerance: Double = 1E-10

    println(s"Running Generalized CSTF Tree on $inputFile")


    var sparkS: SparkConf = new SparkConf().setAppName("CSTFRunner")
//      .set("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:ConcGCThreads=16 -XX:ParallelGCThreads=16 -XX:InitiatingHeapOccupancyPercent=35 -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
//      .set("spark.executor.extraJavaOptions", "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps")
    //      sparkS = sparkS.setMaster("local[*]")

    var rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)
    val sc = new SparkContext(sparkS)
    rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)

    val outputFile = "CSTF_Output"
    val data: RDD[String] = sc.textFile(inputFile)
    val tensor:RDD[Vector] = CSTFUtils.FileToTensor(data).cache()


    var rt: Double = 0.0
    if (version == 0) {
      println("Running Generalized Tree COO")
      rt = CSTFTreeGeneralized.CP_ALS(tensor, maxIterations, rank, tolerance, sc, outputFile)
    } else if (version == 1) {
      println("Running generalized TensorCP COO")
      rt = TensorCPGeneralized.CP_ALS(tensor, maxIterations, rank, tolerance, sc)
    } else if (version == 2) {
      println("Running generalized QCOO")
      rt = COOGeneralized.CP_ALS(tensor, maxIterations, rank, tolerance, sc)
    } else if (version == 3) {
      println("Running generalized QCOO with native spark ops")
      rt = COOGeneralizedRowMatrix.CP_ALS(tensor, maxIterations, rank, tolerance, sc)
    }
    println(s"Running time: $rt")

  }

}
