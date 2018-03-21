/**
  * Created by cqwcy201101 on 4/28/17.
  */

import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{max, pinv, sum, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.HashPartitioner
import breeze.numerics._

import scala.util.control.Breaks



object CSTFTree {

  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf()
      .setMaster("local[1]")
      .setAppName("CSTF_COO")
      .set("spark.executor.instances", "8")
      .set("spark.executor.cores", "1")


    val sc: SparkContext = new SparkContext(conf)

//    val inputFile = "s3n://tensorcp/TensorCP/Input/" + args.apply(0)
//    val outputFile = "s3n://tensorcp/TensorCP/Output/" + args.apply(1) + "/"
    val inputFile = "random2.txt"
    val outputFile = "CSTF_Output"

    val Data:RDD[String] = sc.textFile(inputFile)
    val TensorRdd:RDD[Vector] = FileToTensor(Data)


    def Num_Itr = args.apply(0).toInt
    def Rank = args.apply(1).toInt
    def tolerance = 1E-10


    CP_ALS(Num_Itr,TensorRdd,Rank,tolerance,sc,outputFile)




  }





  def FileToTensor(lines: RDD[String]):RDD[Vector] = {
    lines.map(line => Vectors.dense(line.split("\t").map(_.toDouble)))
  }

  def RDD_VtoRowMatrix(RddData:RDD[Vector]):RowMatrix =
  {
    val Result:RowMatrix = new RowMatrix(RddData
      .map(t => Vectors.dense(t.toArray)))

    Result

  }



  def RandRowMatrix(Size:Long,
                    Rank:Int,
                    sc:SparkContext): RowMatrix =
  {
    val rowData = RandomRDDs
      .uniformVectorRDD(sc,Size,Rank)
      .map(x => Vectors.dense(BDV.rand[Double](Rank).toArray))
    val matrixRandom: RowMatrix = new RowMatrix(rowData,Size,Rank)

    matrixRandom
  }



  def Indexed_RowMatrix(Size:Long,
                        Rank:Int,
                        sc:SparkContext):IndexedRowMatrix =
  {
    val tempRowMatrix: RowMatrix = RandRowMatrix(Size,Rank,sc)
    val indexed = tempRowMatrix.rows.zipWithIndex()
      .map{case (x,y) => IndexedRow(y, Vectors.dense(x.toArray))}
    val Indexed_M:IndexedRowMatrix = new IndexedRowMatrix (indexed)

    Indexed_M
  }


  def GenM1(SizeOfMatrix:Long,
            Rank:Int,
            sc:SparkContext):IndexedRowMatrix = {

    val M1:IndexedRowMatrix =
      new IndexedRowMatrix(Indexed_RowMatrix(SizeOfMatrix,Rank,sc)
        .rows.map(x =>
        IndexedRow(x.index,Vectors.zeros(Rank))))

    M1
  }


  def K_Product(v:Double,
                DV_1:BDV[Double],
                DV_2:BDV[Double]):BDV[Double] = {

    val Result:BDV[Double] = (DV_1:*DV_2) :*= v

    Result
  }

  def VtoBDV(v:Vector) = BDV[Double](v.toArray)

  def BDVtoV(bdv:BDV[Double]) = Vectors.dense(bdv.toArray)

  def BDMtoMatrix(InputData:BDM[Double]):Matrix = Matrices.dense(InputData.rows, InputData.cols, InputData.data)




  def Compute_MTM_RowMatrix(matrix:IndexedRowMatrix)={
    val mTm = matrix.computeGramianMatrix()
    val MTM:BDM[Double] = new BDM[Double](mTm.numRows, mTm.numCols, mTm.toArray)

    MTM
  }

  def ComputeM2(m1:IndexedRowMatrix,
                m2:IndexedRowMatrix):Matrix = {

    val M1M = Compute_MTM_RowMatrix(m1)
    val M2M = Compute_MTM_RowMatrix(m2)

    val result:Matrix = BDMtoMatrix(pinv(M1M :* M2M))

    result

  }


  def TensorTree(tensor:RDD[Vector],
                 Dim:Int):RDD[(Vector,List[Vector])] = {

    val index_1 = (Dim+1)%3
    val index_2 = (Dim+2)%3
    val Tree = tensor.map(v => (Vectors.dense(v(index_1),v(index_2)), Vectors.dense(v(Dim) , v(3))))
      .combineByKey(List(_),
        (c:List[Vector],v:Vector) => v::c,
        (c1:List[Vector],c2:List[Vector]) => c1:::c2)

    Tree
  }

  def ComputeM1(TreeTensor:RDD[(Vector,List[Vector])],
                m1:IndexedRowMatrix,
                m2:IndexedRowMatrix,
                SizeOfMatrix:Long,
                Rank:Int,
                sc:SparkContext) =
  {

    val Map_m1 = m1.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector)))
    val Map_m2 = m2.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector)))

    val Tensor_1 = TreeTensor
      .map(pair => (pair._1(0).toLong,pair))
//      .partitionBy(new HashPartitioner(TreeTensor.partitions.length))
//      .persist()

    val Join_m1 = Tensor_1
      .join(Map_m1)
      .map(line => (line._2._1._1(1).toLong, (line._2._1._2,line._2._2)))
      .partitionBy(new HashPartitioner(Tensor_1.partitions.length))
      .persist()

    val Join_m2 = Join_m1
      .join(Map_m2)
      .mapValues(v => (v._1._1,v._1._2 :* v._2))
      .values
      .flatMap(pair => pair._1.map(v => (v(0).toLong, pair._2 :*= v(1))))
      .sortByKey()
      .reduceByKey(_+_)

    val M1:IndexedRowMatrix = new IndexedRowMatrix(Join_m2.map(v => IndexedRow(v._1, BDVtoV(v._2))))
    M1

  }

  def UpdateFM(TensorData: RDD[(Vector,List[Vector])],
               m1: IndexedRowMatrix,
               m2: IndexedRowMatrix,
               //Dim: Int,
               SizeOfMatrix: Long,
               Rank: Int,
               sc:SparkContext
              ): IndexedRowMatrix =
  {
    ComputeM1(TensorData,m1,m2,SizeOfMatrix,Rank,sc)
      .multiply(ComputeM2(m1,m2))
  }

  def UpdateLambda(matrix:IndexedRowMatrix,
                   N:Int) =
  {
    if (N == 0)
      VtoBDV(matrix.toRowMatrix().computeColumnSummaryStatistics().normL2)
    else
      VtoBDV(matrix.toRowMatrix().computeColumnSummaryStatistics().max)
  }


  def ComputeFit(TreeTensor:RDD[(Vector,List[Vector])],
                 TensorData:RDD[Vector],
                 L:BDV[Double],
                 A:IndexedRowMatrix,
                 B:IndexedRowMatrix,
                 C:IndexedRowMatrix,
                 ATA:BDM[Double],
                 BTB:BDM[Double],
                 CTC:BDM[Double]) =
  {
    val tmp: BDM[Double] = (L * L.t) :* ATA :* BTB :* CTC
    val normXest = abs(sum(tmp))
    val norm = TensorData.map(x => x.apply(3) * x.apply(3)).reduce(_ + _)

    var product = 0.0
    val Result = TreeTensor
      .map(x => (x._1(0).toLong,x))
      .join(B.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .mapValues(x => (x._1._1(1).toLong,x)).values
      .join(C.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .mapValues(x => (x._1._1,x._1._2:*x._2)).values
      .flatMap(x => x._1._2.map(v => (v(0).toLong, x._2 :*= v.apply(1))))
      .join(A.rows.map(idr => (idr.index.toLong, VtoBDV(idr.vector))))
      .mapValues(v => v._1:*v._2)
      .values
      .reduce(_+_)

    product = product + Result.t * L
    val residue = sqrt(normXest + norm - 2*product)
    val Fit = 1.0 - residue/sqrt(norm)


    Fit
  }

  def NormalizeMatrix(matrix:IndexedRowMatrix,
                      L:BDV[Double]) ={

    val map_M = matrix.rows.map(x =>
      (x.index, BDV[Double](x.vector.toArray))).mapValues(x => x:/L)

    val NL_M:IndexedRowMatrix = new IndexedRowMatrix(map_M
      .map(x =>
        IndexedRow(x._1, Vectors.dense(x._2.toArray))))

    NL_M
  }




  def CP_ALS(IterNum: Int,
             //TreeTensor:RDD[(Vector,List[Vector])],
             TensorData:RDD[Vector],
             Rank:Int,
             Tolerance:Double,
             sc:SparkContext,
             outputPath:String) =
  {
    val loop = new Breaks

    val Tree_CBA = TensorTree(TensorData,0).cache()
    val Tree_CAB = TensorTree(TensorData,1).cache()
    val Tree_ABC = TensorTree(TensorData,2).cache()


    val SizeVector = RDD_VtoRowMatrix(TensorData).computeColumnSummaryStatistics().max
    val OrderSize = List(SizeVector(0).toLong+1,SizeVector(1).toLong+1,SizeVector(2).toLong+1)

    var MA = Indexed_RowMatrix(OrderSize(0),2,sc)
    var MB = Indexed_RowMatrix(OrderSize(1),2,sc)
    var MC = Indexed_RowMatrix(OrderSize(2),2,sc)
    var Lambda: BDV[Double] = BDV.zeros(Rank)


    var fit = 0.0
    var pre_fit = 0.0
    var val_fit = 0.0
    var N:Int = 1

    def Update_NFM(TreeTensor:RDD[(Vector,List[Vector])] ,
                   m1:IndexedRowMatrix,
                   m2:IndexedRowMatrix,
                   Size:Long,
                   N:Int) =
    {
      var M = UpdateFM(TreeTensor,m1,m2,Size,Rank,sc)
      Lambda= UpdateLambda(M,N)
      M = NormalizeMatrix(M,Lambda)

      M
    }

    val time_s:Double=System.nanoTime()
    loop.breakable
    {
      for (i <- 0 until IterNum)
      {
        MA = Update_NFM(Tree_CBA,MB,MC,MA.numRows(),i)
        MB = Update_NFM(Tree_CAB,MC,MA,MB.numRows(),i)
        MC = Update_NFM(Tree_ABC,MA,MB,MC.numRows(),i)

        pre_fit = fit
        fit = ComputeFit (
          Tree_CBA,
          TensorData,
          Lambda,
          MA,
          MB,
          MC,
          Compute_MTM_RowMatrix(MA),
          Compute_MTM_RowMatrix(MB),
          Compute_MTM_RowMatrix(MC)
        )
        val_fit = abs(fit - pre_fit)
        N = N+1

        if (val_fit<Tolerance)
          loop.break
      }
    }
    val time_e:Double=System.nanoTime()
    val runtime = (time_e - time_s)/1000000000 + "s"

    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")
    println("fit = " + val_fit)
    println("Iteration times = " + N)

    while (true) {
      Thread.sleep(100)
    }
    val RDDTIME = sc.parallelize(List(runtime))

//    RDDTIME.distinct().repartition(1).saveAsTextFile(outputPath)




  }




}
