package org.paramath
//import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SparkTestSuite extends FunSuite with BeforeAndAfterAll{

//  @transient var spark: SparkSession = _
  @transient var spark: SparkConf = _
  @transient var sc: SparkContext = _
  @transient var checkpointDir: String = _
  override def beforeAll() {

//    spark = SparkSession.builder
//      .master("local[8]")
//      .appName("CSTF_UnitTest")
//      .getOrCreate()
//    val rl = Logger.getRootLogger()
//    rl.setLevel(Level.ERROR)
//    sc = spark.sparkContext

    spark = new SparkConf()
      .setMaster("local[8]")
      .setAppName("CSTF_UnitTest")
    val rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)
    sc = new SparkContext(spark)
  }

}