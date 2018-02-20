package org.paramath

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import org.apache.log4j.{Level, Logger}

class SparkTestSuite extends FunSuite with BeforeAndAfterAll{

  @transient var spark: SparkSession = _
  @transient var sc: SparkContext = _
  @transient var checkpointDir: String = _
  override def beforeAll() {

    spark = SparkSession.builder
      .master("local[8]")
      .appName("CSTF_UnitTest")
      .getOrCreate()
    val rl = Logger.getRootLogger()
    rl.setLevel(Level.ERROR)
    sc = spark.sparkContext
  }

}