package org.paramath.CSTF

/**
  * Generalized COO Tensor Implementation for 3+ order tensors.
  */

import breeze.linalg.{pinv, sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.{abs, sqrt}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.paramath.CSTF.utils.CSTFUtils
import org.paramath.structures.IRowMatrix

import scala.collection.mutable.{ListBuffer, Queue}
import scala.util.control.Breaks

object COOGeneralizedRowMatrix {


  /**
    * Perform the CP_ALS decomposition
    * @return The decomposition of the tensor.
    */
  def CP_ALS(tensorData: RDD[Vector],
             maxIterations: Int,
             rank: Int,
             tolerance: Double,
             sc: SparkContext): Double = {

    val sizeVector = CSTFUtils.RDD_DVtoRowMatrix(tensorData)
      .computeColumnSummaryStatistics().max
    val normVal = tensorData.map(x => x(x.size-1)*x(x.size-1)).reduce(_+_)

    val dims = tensorData.first().size - 1

    val time_s: Long = System.nanoTime()
    val maxDimSizes: ListBuffer[Long] = ListBuffer[Long]()
    for(i <- 0 until sizeVector.size - 1) {
      maxDimSizes += (sizeVector(i) + 1).toLong
    }

    val matrices = new Array[IndexedRowMatrix](dims)
    for(i <- 0 until matrices.length) {
      matrices(i) = CSTFUtils.randomIndexedRowMatrix(maxDimSizes(i), rank, sc)
    }

    var N:Int = 0

    var lambda:BDV[Double] = BDV.zeros(rank)
    var fit:Double = 0.0
    var prev_fit:Double = 0.0
    var val_fit:Double = 0.0


    val loop = new Breaks


    // The queue of calculated gram matrices
    val gmQ: Queue[BDM[Double]] = new Queue[BDM[Double]]()

    // READ THIS TO GET A BETTER UNDERSTANDING OF HOW THIS ALGORITHM WORKS
    // Start with the 0th index of our COO data (i.e. put the vector for V_i in the queue)
    // Once V_i is in the queue add the rest of the vectors to the queue
    var circular: RDD[(Long, (Vector, Queue[Vector]))] = tensorData
      .map(v => (v(0).toLong, v))
      .join(CSTFUtils.splitIndexedRowMatrix(matrices(0))) // join the ith vectors
      .map({
      case(ind: Long, (v: Vector, data: Vector)) => {
        val q = new Queue[Vector]()
        q.enqueue(data)
        (v(1 % dims).toLong, (v, q))
      }
    }).cache()

    for(i <- 1 until matrices.length - 1) {
      circular = circular.join(CSTFUtils.splitIndexedRowMatrix(matrices(i)))
        .map({ case(ind: Long, ((vec: Vector, vQ: Queue[Vector]), vNew: Vector)) => {
        vQ.enqueue(vNew)
        (vec((i + 1) % dims).toLong, (vec, vQ))
      }})
//      val imd = (i+1) % dims
//      println(s" i:$i, (i+1)%dims: $imd")
    }

    // Compute ALL of the gram matrices and add them to the queue
    for(i <- 0 until matrices.length) {
      gmQ.enqueue(CSTFUtils.MatrixToBDM(matrices(i).computeGramianMatrix()))
    }

    loop.breakable{

      for (i <- 0 until maxIterations)
      {
        var tick = System.currentTimeMillis()
        val cpalstick = System.currentTimeMillis()
        var tock = System.currentTimeMillis()


        // Compute the updates for all matrices
        for(j <- 0 until matrices.length) {
          tick = System.currentTimeMillis()

          // Get the matrix which we need to add to join on
          val matInd: Int = if(j == 0) matrices.length - 1 else ( (j-1) % matrices.length)
          val mat: IndexedRowMatrix = matrices(matInd)
          // =================================================================
          // =================== MTTKRP OPERATION ============================
          // =================================================================

          circular = circular.join(CSTFUtils.splitIndexedRowMatrix(mat)).map({
            case (oldInd: Long, ((vec: Vector, q: Queue[Vector]), vNew: Vector)) => {
              q.dequeue()
              q.enqueue(vNew)
              (vec(j).toLong, (vec, q))
            }
          }).cache()
          val matRows = circular.mapValues({
            case (vec: Vector, q: Queue[Vector]) => {
              val vNew: BDV[Double] = q.map(v => CSTFUtils.VtoBDV(v)).reduce(_ :* _)
              vNew :* vec(dims)
            }
          }).reduceByKey(_ + _)
          // =================================================================
          // =================== MTTKRP OPERATION ============================
          // =================================================================

          // MTTKRP Complete
          // This is the first matrix of the CP_ALS algorithm
          val M1 = new IndexedRowMatrix(matRows.map(f => IndexedRow(f._1, CSTFUtils.BDVtoV(f._2))))

          // Compute M2 - All of the gram matrices together
          // Use the queue of GM's to calculate the product
          // currently the queue should contain ALL of the gram matrices for each dimension.
          // Dequeue (our current dimension) and then multiply all of the matrices together elementwise
          gmQ.dequeue()
          val M2: BDM[Double] = pinv(gmQ.reduce((m1, m2) => m1 :* m2))

          matrices(j) = M1.multiply(CSTFUtils.BDMToMatrix(M2))
          matrices(j).rows.cache()

          lambda = CSTFUtils.updateLambda(matrices(j), i)
          matrices(j) = CSTFUtils.normalizeMatrix(matrices(j), lambda)

          // Enqueue the new gram matrix
          gmQ.enqueue(CSTFUtils.MatrixToBDM(matrices(j).computeGramianMatrix()))


          tock = System.currentTimeMillis()
          CSTFUtils.printTime(tick, tock, s"Compute M$j $i")

        }

        val cpalstock = System.currentTimeMillis()
        CSTFUtils.printTime(cpalstick, cpalstock, s"CP_ALS $i")
        prev_fit = fit
        tick = System.currentTimeMillis()
        fit = computeFitCircular(circular, matrices.last, gmQ, lambda, normVal)
//        fit = computeFit(tensorData, lambda, matrices)
        tock = System.currentTimeMillis()
        CSTFUtils.printTime(tick, tock, s"Compute Fit $i")
        val_fit = abs(fit - prev_fit)
        val ttime: Double = ((cpalstock-cpalstick)/1000.0) + ((tock-tick)/1000.0)
        println()
        println(s"Total CP_ALS $i $ttime")
        println(s"Fit Value $i : $val_fit")


        N = N +1

        if (val_fit < tolerance)
          loop.break
      }


    }

    //---------------
    val time_e:Double=System.nanoTime()

    //---------------

    println(val_fit)
    println(N)
    println(lambda)

    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")
    0.0
  }


  /**
    *
    * @param circularRDD
    * @param jMat
    * @param gms
    * @param lambda
    * @param normVal
    * @return
    */
  def computeFitCircular(circularRDD: RDD[(Long, (Vector, Queue[BDV[Double]]))],
                         jMat: IRowMatrix,
                         gms: Queue[BDM[Double]],
                         lambda: BDV[Double],
                         normVal: Double): Double = {
    val tmp = (lambda * lambda.t) :* gms.reduce((m1, m2) => m1 :* m2)
    val normXest = abs(sum(tmp))
    var product = 0.0


    val result: BDV[Double] = circularRDD.join(jMat.rows).map({
      case (oldInd: Long, ((v: Vector, q: Queue[BDV[Double]]), vNew: BDV[Double])) => {
        (vNew :* q.reduce(_ :* _)) :* v(v.size-1)
      }
    }).reduce(_ + _)
    product += (result.t * lambda)
    val residue = sqrt(normXest + normVal - (2 * product))
    val fit = 1.0 - (residue/sqrt(normVal))
    fit


  }
  def computeFitCircular(circularRDD: RDD[(Long, (Vector, Queue[Vector]))],
                         jMat: IndexedRowMatrix,
                         gms: Queue[BDM[Double]],
                         lambda: BDV[Double],
                         normVal: Double): Double = {
    val tmp = (lambda * lambda.t) :* gms.reduce((m1, m2) => m1 :* m2)
    val normXest = abs(sum(tmp))
    var product = 0.0


    val result: BDV[Double] = circularRDD.join(CSTFUtils.splitIndexedRowMatrix(jMat)).map({
      case (oldInd: Long, ((v: Vector, q: Queue[Vector]), vNew: Vector)) => {
        (CSTFUtils.VtoBDV(vNew) :* q.map(v => CSTFUtils.VtoBDV(v)).reduce(_ :* _) :* v(v.size-1))
      }
    }).reduce(_ + _)
    product += (result.t * lambda)
    val residue = sqrt(normXest + normVal - (2 * product))
    val fit = 1.0 - (residue/sqrt(normVal))
    fit


  }

  def computeFit(tensor: RDD[Vector],
                 lambda: BDV[Double],
                 mats: Array[IRowMatrix]): Double = {
    var tmp:BDM[Double] = (lambda * lambda.t)
    for (i <- 0 until mats.length) {
      tmp :*= mats(i).computeGramian()
    }
    val normXest = abs(sum(tmp))
    val dims = tensor.first().size - 1
    val norm = tensor.map(x => x(dims)*x(dims)).reduce(_+_)

    var product = 0.0

    var joinedVecs: RDD[(Long, (Vector, BDV[Double]))] = tensor.map(v => (v(0).toLong, v))
      .join(mats(0).rows)
      .map(f => (f._2._1(1).toLong, (f._2)))
    for(i <- 1 until mats.length) {
      joinedVecs = joinedVecs
        .join(mats(i).rows)
        .map({
          case(ind: Long, ((v: Vector, vOld: BDV[Double]), vNew: BDV[Double])) => {
            (v(i+1).toLong, (v, vOld :* vNew))
          }})
    }
    val t1: RDD[(BDV[Double])] = joinedVecs.map({
      case(i: Long, (v: Vector, vOld: BDV[Double])) => {
        vOld :* v(v.size-1)
      }})
    val result = t1.reduce(_ + _)
    product += (result.t * lambda)
    val residue = sqrt(normXest + norm - (2 * product))
    val fit = 1.0 - (residue/sqrt(norm))
    fit
  }

  /**
    * Drops the i-th index from the array and rotates the array so that i+1
    * in the original array is now the 0th index and i-1 is the last index.
    * @param v - The original array
    * @param i - Index to drop and rotate at
    * @return
    */
  def dropAndRotate(v: Array[IRowMatrix], i: Int): Array[IRowMatrix] = {
    val sz = v.length
    val (first, last) = v.splitAt(i % sz)
    (last ++ first).slice(1, v.length)
  }

  /**
    *
    * @param tensor Original tensor data
    * @param mats The matrices to join on
    * @param dim dimension of the vector to perform MTTKRP on
    * @param numRows - number of rows in resulting matrix.
    * @param rank Rank of resulting matrix
    * @param sc spark context
    * @return The result of MTTKRP
    */
  def mttkrp(tensor: RDD[Vector],
             mats: Array[IRowMatrix],
             dim: Int,
             numRows: Long,
             rank: Int,
             sc: SparkContext): IRowMatrix = {
    // Assume matrices in array are passed in correct order
//    val InitialM1: IndexedRowMatrix = IRowMatrix.zeros(numRows,rank,sc)



    // Indexes from 0 to size - 2 ::: v(size-1) is the tensor value, v(dim) is the dim we don't join on
    // i.e. 3rd order tensor will have vector size 4
    // Thus our indexes should range from 0 to 2 during mttkrp
    val totalDims: Int = tensor.first().size - 1 // when using mod (%) gives values between [0, 2]
    var tmp: RDD[(Long, (Vector, BDV[Double]))] = tensor
      .map(v => (v( (dim+1) % totalDims ).toLong, v))
      .join(mats(0).rows) // join the 0th matrix
      .map({
        case(ind: Long, (v: Vector, data: BDV[Double])) => (v((dim + 2) % totalDims).toLong, (v, data))
      })

    for(i <- 1 until mats.length - 1) {
      val t1 = tmp.join(mats(i).rows)
      tmp = t1.map({ case(ind: Long, ((vec: Vector, vOld: BDV[Double]), vNew: BDV[Double])) => {
        (vec( ((dim+1)+ i + 1) % totalDims).toLong, (vec, vOld :* vNew))
      }})
    }
    val result: RDD[(Long, BDV[Double])] = tmp.join(mats(mats.length-1).rows)
      .map({ case(ind: Long, ((v: Vector, vOld: BDV[Double]), vNew: BDV[Double])) => {
          (v(dim).toLong, vOld :* vNew :* v(totalDims))
        }
      }).reduceByKey(_ + _)

//    val tempM1: RDD[(Long, Vector)] = InitialM1.rows.map(
//          x => (x.index, vecToBDV(x.vector) ))
//          .cogroup(result)
//          .mapValues{x =>
//            if (x._2.isEmpty) {
//              BDVtoVector(x._1.head)
//            }  else {
//              BDVtoVector(x._2.head)
//            }
//          }

    new IRowMatrix(result)
  }


  def computeM2(mats: Array[IRowMatrix]): BDM[Double] = {
    var x1 = mats(0).computeGramian()
    for(i <- 1 until mats.length) {
      x1 = x1 :* mats(i).computeGramian()
    }
    pinv(x1)
  }

  def mttkrpProduct(tensor: RDD[Vector],
                    mats: Array[IRowMatrix],
                    rank: Int,
                    dim: Int,
                    nRows: Long,
                    sc:SparkContext): IRowMatrix =
  {
    mttkrp(tensor, mats, dim, nRows, rank, sc).multiply(computeM2(mats))
  }
}


