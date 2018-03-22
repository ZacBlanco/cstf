package org.paramath.CSTF

/**
  * Created by cqwcy201101 on 5/4/17.
  */

import breeze.linalg.{pinv, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkConf, SparkContext}
import org.paramath.CSTF.utils.CSTFUtils
import org.paramath.structures.IRowMatrix
import scala.util.control.Breaks


object CSTFCOO {

  def main(args: Array[String]): Unit = {

    val conf: SparkConf = new SparkConf().setAppName("CSTF_COO")

    val sc: SparkContext = new SparkContext(conf)
    //

    val inputFile = args(0)
    val outputFile = " CSTF_COO_Output"


    def Num_Itr = args.apply(1).toInt

    def Rank = args(2).toInt

    def Num_nodes = args(3).toInt

    val Data = sc.textFile(inputFile)
    val TensorRdd = CSTFUtils.FileToTensor(Data)


    def tolerance = 1E-10

    CP_ALS(Num_Itr, TensorRdd, Rank, tolerance, sc, outputFile, Num_nodes)


  }

  //
  //  def FileToTensor(lines: RDD[String]): RDD[Vector] =
  //  {
  //    //ines.map(line => Vectors.dense(line.split("\t").map(_.toDouble)))
  //
  //    lines.map(line => line.split("\t").map(_.toDouble)).zipWithIndex()
  //      .map(x => Vectors.dense(x._1.toList.::(x._2.toDouble).toArray))
  //  }


  def FileToTensor(lines: RDD[String]): RDD[Vector] = {
    lines.map(line => Vectors.dense(line.split("\t").map(_.toDouble)))

  }


  def RDD_VtoRowMatrix(RddData: RDD[Vector]): RowMatrix = {
    val Result: RowMatrix = new RowMatrix(RddData
      .map(t => Vectors.dense(t.toArray)))

    Result
  }


  def RandRowMatrix(Size: Long,
                    Rank: Int,
                    sc: SparkContext): RowMatrix = {
    val rowData = RandomRDDs
      .uniformVectorRDD(sc, Size, Rank)
      .map(x => Vectors.dense(BDV.rand[Double](Rank).toArray))
    val matrixRandom: RowMatrix = new RowMatrix(rowData, Size, Rank)

    matrixRandom
  }


  def ToIndexRM(rdd: RDD[(Long, BDV[Double])]) =
    new IndexedRowMatrix(rdd.map(pair => IndexedRow(pair._1, BDVtoV(pair._2))))


  def Indexed_RowMatrix(Size: Long,
                        Rank: Int,
                        sc: SparkContext,
                        N: Int): IndexedRowMatrix = {
    val tempRowMatrix: RowMatrix = RandRowMatrix(Size, Rank, sc)
    val indexed = tempRowMatrix.rows.zipWithIndex()
      .map { case (x, y) => (y, Vectors.dense(x.toArray)) }
      .partitionBy(new MyPartitioner(32 * (N - 1))).persist()

    new IndexedRowMatrix(indexed.map(pair => IndexedRow(pair._1, pair._2)))
  }

  def GenM1(SizeOfMatrix: Long,
            Rank: Int,
            sc: SparkContext,
            N: Int): IndexedRowMatrix = {

    val M1: IndexedRowMatrix =
      new IndexedRowMatrix(Indexed_RowMatrix(SizeOfMatrix, Rank, sc, N)
        .rows.map(x =>
        IndexedRow(x.index, Vectors.zeros(Rank))))

    M1
  }

  def K_Product(v: Double,
                DV_1: BDV[Double],
                DV_2: BDV[Double]): BDV[Double] = {

    val Result: BDV[Double] = (DV_1 :* DV_2) :*= v

    Result
  }


  def VtoBDV(v: Vector) =
    BDV[Double](v.toArray)


  def BDVtoV(bdv: BDV[Double]) =
    Vectors.dense(bdv.toArray)


  def BDMtoMatrix(InputData: BDM[Double]): Matrix =
    Matrices.dense(InputData.rows, InputData.cols, InputData.data)


  def Compute_MTM_RowMatrix(matrix: IndexedRowMatrix) = {
    val mTm = matrix.computeGramianMatrix()
    val MTM: BDM[Double] = new BDM[Double](mTm.numRows, mTm.numCols, mTm.toArray)

    MTM
  }

  def ComputeM2(m1: IndexedRowMatrix,
                m2: IndexedRowMatrix
               ): BDM[Double] = {

    val M1M = Compute_MTM_RowMatrix(m1)
    val M2M = Compute_MTM_RowMatrix(m1)

    var x1 = M1M :* M2M
    var singular: Boolean = false
    try{
      x1 = breeze.linalg.inv(x1)
    } catch {
      case mse: breeze.linalg.MatrixSingularException => singular = true
    }
    if (singular) {
      x1 = breeze.linalg.pinv(x1)
    }

    x1
//    val result: Matrix = BDMtoMatrix(pinv(M1M :* M2M))
//    result

  }

  def PairRDD(Data: RDD[Vector],
              Dim: Int,
              N: Int) = {
    //val n = Data.partitions.length
    val mapped = Data.map(line => (line(Dim).toLong, line))
    mapped.partitionBy(new MyPartitioner(32 * (N - 1))).persist(StorageLevel.MEMORY_AND_DISK)
    //val partitioner = new HashPartitioner(Data.partitions.length)
    //rtitionBy(partitioner).persist()
  }

  def UpdateM(M: IndexedRowMatrix,
              m1: RDD[(Long, BDV[Double])],
              m2: Matrix,
              Rank: Int
             ) = {

    //    ToIndexRM( IRMtoRdd(M).join(m1).mapValues(f => (VtoBDV(f._1), VtoBDV(f._2))))
    ToIndexRM(IRMtoRdd(M).leftOuterJoin(m1)
      .mapValues(pair => {
        if (pair._2.nonEmpty) pair._2.get else VtoBDV(Vectors.zeros(Rank))
      }))
      .multiply(m2)




    //val tmp = M.leftOuterJoin(m1)
    //  .mapValues(pair => {if (pair._2.nonEmpty) pair._2.get else pair._1})
    //.collect()

    //val tmp = M.leftOuterJoin(m1).mapValues(pair => pair._2.get)
    //   .map(pair => IndexedRow(pair._1,BDVtoV(pair._2)))


    /*
    ToIndexRM (
     M.leftOuterJoin(m1)
     .mapValues(pair => {if (pair._2.nonEmpty) pair._2.get else pair._1})
   )
     .multiply(m2)
     .rows
     .map(idr => (idr.index,VtoBDV(idr.vector)))

    //M.leftOuterJoin(m1).mapValues(pair => {if (pair._2.nonEmpty) pair._2.get else pair._1})
*/


  }

  def IRMtoRdd(IdexRM: IndexedRowMatrix) =
    IdexRM.rows.map(idr => (idr.index, VtoBDV(idr.vector)))


  def ComputeM1(PairRdd1: RDD[(Long, Vector)],
                m1: IndexedRowMatrix,
                m2: IndexedRowMatrix,
                Index: Int, //index to update (0,1,2)
                //SizeOfMatrix:Long,
                Rank: Int) = {

    //val Init_M1:IndexedRowMatrix = GenM1(SizeOfMatrix,Rank,sc)

    //val Map_m1 = m1.map(pair => (idr.index, VtoBDV(idr.vector)))
    //val Map_m2 = m2.map(idr => (idr.index, VtoBDV(idr.vector)))
    /*
          val Join_m1 = PairRdd1.join(m1)
            .map(line => (line._2._1(Index).toLong,line._2._1(3)*line._2._2))

          val Join_m2 = PairRdd2.join(m2)
            .map(line => (line._2._1(Index).toLong,line._2._2))

          val M1 = Join_m1.join(Join_m2)
            .mapValues(v => v._1:*v._2)
            .reduceByKey(_+_)

     */
    val index_1 = (Index + 1) % 3
    val index_2 = (Index + 2) % 3

    val M1 = PairRdd1.join(IRMtoRdd(m1))
      .map { case (long, (vector, bdv)) => (vector(index_2).toLong, (vector, vector(3) * bdv)) }
      .join(IRMtoRdd(m2))
      .map { case (long, ((vector, bdv1), bdv2)) => (vector(Index).toLong, bdv1 :* bdv2) }
      .reduceByKey(_ + _)

    M1


    // val tmp2 = PairRdd2.join(IRMtoRdd(m2)).map{case (long,(vector,bdv)) => (vector(0), (vector(Index).toLong, bdv))}


    // val M1 = tmp1.join(tmp2)
    //   .map{case (rowNum,((outputIdx1, bdv1),(outputIdx2, bdv2))) => (outputIdx1, bdv1:*bdv2)}
    //   .reduceByKey(_+_)


    //.map(pair => (pair._2._1._1, pair._2._1._2:*pair._2._2._2))
    //.reduceByKey(_+_)


    //.map(line => (line._2._1(Index).toLong,line._2._1(3)*line._2._2))
    //.join(PairRdd2.leftOuterJoin(IRMtoRdd(m2))
    //.map(line => (line._2._1(Index).toLong,line._2._2)))

  }

  /*
    def UpdateFM(PairRdd1:RDD[(Long,Vector)],
                 PairRdd2:RDD[(Long,Vector)],
                 m1:RDD[(Long,BDV[Double])],
                 m2:RDD[(Long,BDV[Double])],
                 Index: Int,  //index to update (0,1,2)
                 SizeOfMatrix:Long,
                 Rank:Int
                 //,
                 //sc:SparkContext
                ) = {

      val M1 = new IndexedRowMatrix(ComputeM1(PairRdd1,PairRdd2,m1,m2,Index,SizeOfMatrix,Rank)
        .map(pair => IndexedRow(pair._1, BDVtoV(pair._2)) ))

      M1.multiply(ComputeM2(m1,m2)).rows.map(idr => (idr.index,VtoBDV(idr.vector)))
    }

  */

  def UpdateLambda(matrix: IndexedRowMatrix,
                   N: Int) = {
    if (N == 0)
      VtoBDV(matrix.toRowMatrix().computeColumnSummaryStatistics().normL2)
    else
      VtoBDV(matrix.toRowMatrix().computeColumnSummaryStatistics().max)
  }


  def ComputeFit(TensorData: RDD[Vector],
                 PairRdd1: RDD[(Long, Vector)],
                 PairRdd2: RDD[(Long, Vector)],
                 PairRdd3: RDD[(Long, Vector)],
                 L: BDV[Double],
                 A: IndexedRowMatrix,
                 B: IndexedRowMatrix,
                 C: IndexedRowMatrix) = {
    val ATA = Compute_MTM_RowMatrix(A)
    val BTB = Compute_MTM_RowMatrix(B)
    val CTC = Compute_MTM_RowMatrix(C)

    val tmp: BDM[Double] = (L * L.t) :* ATA :* BTB :* CTC
    val normXest = abs(sum(tmp))
    val norm = TensorData.map(x => x(3) * x(3)).reduce(_ + _)

    val Joined_1 = PairRdd1
      .join(IRMtoRdd(A))
      .map(pair => (pair._2._1(0), pair._2._1(3) * pair._2._2))

    val Joined_2 = PairRdd2
      .join(IRMtoRdd(B))
      .map(pair => (pair._2._1(0), pair._2._2))

    val Joined_3 = PairRdd3
      .join(IRMtoRdd(C))
      .map(pair => (pair._2._1(0), pair._2._2))

    val Result = Joined_1.join(Joined_2).join(Joined_3)
      .mapValues(x => x._1._1 :* x._1._2 :* x._2)
      .values
      .reduce(_ + _)

    val product = Result.t * L
    val residue = sqrt(normXest + norm - 2 * product)
    val Fit = 1.0 - residue / sqrt(norm)

    Fit


  }

  def NormalizeMatrix(matrix: IndexedRowMatrix,
                      L: BDV[Double]) = {

    val NM = matrix.rows.map(idr => (idr.index, VtoBDV(idr.vector) :/ L))
    ToIndexRM(NM)
  }

  def IndexedRowMatrixToIRM(m1: IndexedRowMatrix): IRowMatrix = {
    new IRowMatrix(m1.rows.map(r => (r.index, VtoBDV(r.vector))))
  }


  def CP_ALS(IterNum: Int,
             TensorData: RDD[Vector],
             Rank: Int,
             tolerance: Double,
             sc: SparkContext,
             outputPath: String,
             Num_node: Int
            ) = {
    val loop = new Breaks
    var fit = 0.0
    var pre_fit = 0.0
    var val_fit = 0.0
    var N: Int = 0
    var Lambda: BDV[Double] = BDV.zeros(Rank)

    val Indx_i = 0
    val Indx_j = 1
    val Indx_k = 2

    val num_tensor = TensorData.partitions.length


    val SizeVector = RDD_VtoRowMatrix(TensorData).computeColumnSummaryStatistics().max
    val OrderSize = Vector(SizeVector(Indx_i).toLong + 1, SizeVector(Indx_j).toLong + 1, SizeVector(Indx_k).toLong + 1)

    val cfTree: RDD[(Vector, List[Vector])] = CSTFUtils.TensorTree(TensorData, 0)
    var tick = System.currentTimeMillis()
    var tock = System.currentTimeMillis()
    var MA: IndexedRowMatrix = Indexed_RowMatrix(OrderSize(Indx_i), Rank, sc, Num_node)
    var MB: IndexedRowMatrix = Indexed_RowMatrix(OrderSize(Indx_j), Rank, sc, Num_node)
    var MC: IndexedRowMatrix = Indexed_RowMatrix(OrderSize(Indx_j), Rank, sc, Num_node)


    val pairRdd_I = PairRDD(TensorData, Indx_i, Num_node)
    val pairRdd_J = PairRDD(TensorData, Indx_j, Num_node)
    val pairRdd_K = PairRDD(TensorData, Indx_k, Num_node)

    val num_pair = pairRdd_I.partitions.length


    def Update_NFM(M: IndexedRowMatrix,
                   PairRdd: RDD[(Long, Vector)],
                   m2: IndexedRowMatrix,
                   m3: IndexedRowMatrix,
                   Index: Int,
                   N: Int) = {
      val t1 = new IRowMatrix(ComputeM1(PairRdd, m2, m3, Index, Rank))
      val t2 = ComputeM2(m2, m3)
      val tmpFM = t1.multiply(t2)
      Lambda = CSTFUtils.updateLambda(tmpFM, N)
      val a = CSTFUtils.normalizeMatrix(tmpFM, Lambda)
      new IndexedRowMatrix(a.rows.map(r => IndexedRow(r._1, BDVtoV(r._2))))
    }


    val time_s: Double = System.nanoTime()


    loop.breakable {
      for (i <- 0 until IterNum) {
        val cpalstick = System.currentTimeMillis()
        tick = System.currentTimeMillis()
        MA = Update_NFM(MA, pairRdd_J, MB, MC, 0, Rank)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute MA $i")
        tick = tock
        MB = Update_NFM(MB, pairRdd_K, MC, MA, 1, Rank)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute MB $i")
        tick = tock
        MC = Update_NFM(MC, pairRdd_I, MA, MB, 2, Rank)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute MB $i")
        val cpalstock = System.currentTimeMillis()

        CSTFUtils.printTime(cpalstick, cpalstock, s"CP_ALS $i")

        pre_fit = fit
        tick = System.currentTimeMillis()
        fit = CSTFUtils.computeFit2(cfTree,
          TensorData,
          Lambda,
          IndexedRowMatrixToIRM(MA),
          IndexedRowMatrixToIRM(MB),
          IndexedRowMatrixToIRM(MC),
          Compute_MTM_RowMatrix(MA),
          Compute_MTM_RowMatrix(MB),
          Compute_MTM_RowMatrix(MC))
        //        fit = ComputeFit(TensorData,
        //          pairRdd_I,
        //          pairRdd_J,
        //          pairRdd_K,
        //          Lambda,
        //          MA,
        //          MB,
        //          MC)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute Fit $i")
        val_fit = abs(fit - pre_fit)
        println(s"Fit $i $val_fit")


        //        N = N +1
        if (val_fit < tolerance)
          loop.break

        N = N + 1
      }
    }


    val time_e: Double = System.nanoTime()

    val runtime = (time_e - time_s) / 1000000000 + "s"
    val numA = MA.rows.partitions.length
    val numB = MB.rows.partitions.length
    val numC = MC.rows.partitions.length


    //    println(" time:" + runtime  + "\nNumIter = " + N  + "\nA = " + numA)
    //    println("partitions_num of Rdd = " + TensorData.partitions.length +
    //     "\npartitions_num of PairRdd = " + pairRdd_I.partitions.length +
    //    "\npartitions_num of A = " + MA.rows.partitions.length)
    //
    //    MA.rows.collect().foreach(println)


    val RDDTIME = sc.parallelize(
      List("Runtime = " + runtime,
        "Num_Iter = " + N,
        "partition_num of A = " + numA,
        "partition_num of B = " + numB,
        "partition_num of C = " + numC,
        "partition_pairRdd =  " + num_pair,
        "partition_TensorRdd = " + num_tensor)
    )

    RDDTIME.distinct().repartition(1).saveAsTextFile(outputPath)


    sc.stop()

  }

  class MyPartitioner(numParts: Int) extends Partitioner {

    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
      (key.toString.toLong % numParts).toInt
    }
  }


}