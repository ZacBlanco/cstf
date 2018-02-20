package org.paramath.CSTF.utils

import org.paramath.structures.IRowMatrix

import breeze.linalg.{pinv, sum, DenseMatrix => BDM, DenseVector => BDV}
import org.paramath.BLAS
import breeze.numerics.{abs, sqrt}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkContext}

object CSTFUtils {

  /**
    * Reads a Tensor file and converts it into an RDD[DenseVector]
    *
    * @param lines
    * @return
    */
  def FileToTensor(lines: RDD[String]): RDD[Vector] = {
    lines.map(line => Vectors.dense(line.split("\t").map(_.toDouble)))
  }

  /**
    * Convert an RDD of DenseVector into a RowMatrix
    *
    * @param RddData
    * @return
    */
  def RDD_DVtoRowMatrix(RddData: RDD[Vector]): RowMatrix = {
    val Result: RowMatrix = new RowMatrix(RddData
      .map(t => Vectors.dense(t.toArray)))

    Result

  }

  /**
    * Generate a random row matrix. where the matrix of of
    *
    * @param Size
    * @param Rank
    * @param sc
    * @return
    */
  def RandRowMatrix(Size: Long,
                    Rank: Int,
                    sc: SparkContext): RDD[BDV[Double]] = {
    RandomRDDs
      .uniformVectorRDD(sc, Size, Rank)
      .map(x => BDV.rand[Double](Rank))
  }


  /**
    * Generate a randomized row matrix and convert it into an indexed row matrix.
    *
    * @param size
    * @param rank
    * @param sc
    * @return
    */
  def Randomized_IRM(size: Long,
                     rank: Int,
                     sc: SparkContext): IRowMatrix = {
    val tempRowMatrix: RDD[BDV[Double]] = RandRowMatrix(size, rank, sc)
    val indexed = tempRowMatrix.zipWithIndex()
      .map { case (x, y) => (y, x) }
    new IRowMatrix(indexed, size,  rank)
  }


  def GenM1(SizeOfMatrix: Long,
            rank: Int,
            sc: SparkContext): IRowMatrix = {



    new IRowMatrix(
      Randomized_IRM(SizeOfMatrix, rank, sc).rows.map( f => (f._1, BDV.zeros[Double](rank)))
    )
  }

  /**
    * Performs the Khatri-Rao Product of two vectors.
    *
    * @param v
    * @param DV_1
    * @param DV_2
    * @return
    */
  def KhatriRao_Product(v: Double,
                        DV_1: BDV[Double],
                        DV_2: BDV[Double]): BDV[Double] = {

    val Result: BDV[Double] = (DV_1 :* DV_2) :*= v

    Result
  }

  def VtoBDV(v: Vector) = BDV[Double](v.toArray)

  def BDVtoV(bdv: BDV[Double]) = Vectors.dense(bdv.toArray)

  def BDMtoMatrix(InputData: BDM[Double]): Matrix = Matrices.dense(InputData.rows, InputData.cols, InputData.data)

  /**
    *
    * Computes the gramian of the matrix (m^TM)
    * @param matrix
    * @return
    */
  def Compute_MTM_RowMatrix(matrix: IRowMatrix): BDM[Double] = {
    matrix.computeGramian()
//    val MTM: BDM[Double] = new BDM[Double](mTm.numRows, mTm.numCols, mTm.toArray)
//
//    MTM
  }


  def ComputeM2(m1: IRowMatrix,
                m2: IRowMatrix): BDM[Double] = {

//    val M1M = Compute_MTM_RowMatrix(m1)
//    val M2M = Compute_MTM_RowMatrix(m2)
//
//    val result: Matrix = BDMtoMatrix(pinv(M1M :* M2M))
//
//    result

    breeze.linalg.pinv(m1.computeGramian() :* m2.computeGramian())
  }

  def UpdateFM(TensorData: RDD[(Vector,List[Vector])],
               m1: IRowMatrix,
               m2: IRowMatrix,
               //Dim: Int,
               SizeOfMatrix: Long,
               rank: Int,
               sc:SparkContext
              ): IRowMatrix =
  {
    ComputeM1(TensorData,m1,m2,SizeOfMatrix,rank,sc)
      .multiply(ComputeM2(m1,m2))
  }



  /**
    * Creates a tensor tree from an RDD of tensors
    *
    * @param tensor
    * @param Dim
    * @return
    */
  def TensorTree(tensor: RDD[Vector],
                 Dim: Int): RDD[(Vector, List[Vector])] = {

    val index_1 = (Dim + 1) % 3
    val index_2 = (Dim + 2) % 3
    val Tree = tensor.map(v => (Vectors.dense(v(index_1), v(index_2)), Vectors.dense(v(Dim), v(3))))
      .combineByKey(List(_),
        (c: List[Vector], v: Vector) => v :: c,
        (c1: List[Vector], c2: List[Vector]) => c1 ::: c2)

    Tree
  }

  def ComputeM1(TreeTensor: RDD[(Vector, List[Vector])],
                m1: IRowMatrix,
                m2: IRowMatrix,
                SizeOfMatrix: Long,
                Rank: Int,
                sc: SparkContext): IRowMatrix = {

    val Map_m1 = m1.rows //m1.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector)))
    val Map_m2 = m2.rows //.map(idr => (idr.index.toLong, VtoBDV(idr.vector)))
    val Init_M1: IRowMatrix = GenM1(SizeOfMatrix, Rank, sc)

    val Tensor_1 = TreeTensor
      .map(pair => (pair._1(0).toLong, pair))
      .partitionBy(new HashPartitioner(TreeTensor.partitions.length))
      .persist()

    val Join_m1 = Tensor_1
      .join(Map_m1)
      .map(line => (line._2._1._1(1).toLong, (line._2._1._2, line._2._2)))
      .partitionBy(new HashPartitioner(Tensor_1.partitions.length))
      .persist()

    val Join_m2 = Join_m1
      .join(Map_m2)
      .mapValues(v => (v._1._1, v._1._2 :* v._2))
      .values
      .flatMap(pair => pair._1.map(v => (v(0).toLong, pair._2 :*= v(1))))
      .sortByKey()
      .reduceByKey(_ + _)

     new IRowMatrix(Join_m2)
  }

  def UpdateLambda(matrix: IRowMatrix,
                   N: Int): BDV[Double] = {
    if (N == 0)
      VtoBDV(matrix.computeColumnSummaryStatistics().normL2)
    else
      VtoBDV(matrix.computeColumnSummaryStatistics().max)
  }


  def ComputeFit(TreeTensor: RDD[(Vector, List[Vector])],
                 TensorData: RDD[Vector],
                 L: BDV[Double],
                 A: IRowMatrix,
                 B: IRowMatrix,
                 C: IRowMatrix,
                 ATA: BDM[Double],
                 BTB: BDM[Double],
                 CTC: BDM[Double]) = {
    val tmp: BDM[Double] = (L * L.t) :* ATA :* BTB :* CTC
    val normXest = abs(sum(tmp))
    val norm = TensorData.map(x => x.apply(3) * x.apply(3)).reduce(_ + _)

    var product = 0.0
    val Result = TreeTensor
      .map(x => (x._1(0).toLong, x))
      // .join(B.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .join(B.rows)
      .mapValues(x => (x._1._1(1).toLong, x)).values
//      .join(C.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .join(C.rows)
      .mapValues(x => (x._1._1, x._1._2 :* x._2)).values
      .flatMap(x => x._1._2.map(v => (v(0).toLong, x._2 :*= v.apply(1))))
//      .join(A.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .join(A.rows)
      .mapValues(v => v._1 :* v._2)
      .values
      .reduce(_ + _)

    product = product + Result.t * L
    val residue = sqrt(normXest + norm - 2 * product)
    val Fit = 1.0 - residue / sqrt(norm)


    Fit
  }

  /**
    * Normalizes matrix values based on the vector L
    *
    * @param matrix
    * @param L
    * @return
    */
  def NormalizeMatrix(matrix: IRowMatrix,
                      L: BDV[Double]): IRowMatrix = {

//        new IndexedRowMatrix(matrix.rows.map(a =>
//          new IndexedRow(a.index, Vectors.dense((BDV[Double](a.vector.toArray) :/ L).toArray)))) // Single line


//    val map_M = matrix.rows.map(x =>
//      (x.index, BDV[Double](x.vector.toArray))).mapValues(x => x :/ L)
//
//    val NL_M: IndexedRowMatrix = new IndexedRowMatrix(map_M
//      .map(x =>
//        IndexedRow(x._1, Vectors.dense(x._2.toArray))))
//
//    NL_M

    new IRowMatrix(
      matrix.rows.mapValues(x => x :/ L)
    )
  }

  def printTime(tick: Long, tock: Long, msg: String): Unit = {
    val time: Long = tock - tick
//    println(s"$msg took $time ms")
  }


}
