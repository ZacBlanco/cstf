package org.paramath.structures



import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.paramath.structures.TT3Util._

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

case class TTEdge(TreeNum: Long, dst: Long)
case class VecInfo(TreeNum: Long, vec: BDV[Double])

class TensorTree3(sc: SparkContext) {
  var data: RDD[Vector] = sc.emptyRDD
  var gr: Array[RDD[(Long, List[TTEdge])]] = new Array[RDD[(Long, List[TTEdge])]](2)
  var imode: RDD[(Long, Double)] = sc.emptyRDD
  var mode = 0
  var dims: Int = -1
  /**
    *
    * @param tensor An RDD of vectors
    * @param mode The mode of the tensor. Between 0 and len(vector) - 1
    * @param sc
    */
  def this(tensor: RDD[Vector], mode: Int, sc:SparkContext) {
    this(sc)
    this.data = tensor
    // Vector structured as i, j, k, value
    val mdv = sc.broadcast(mode)
    this.dims = tensor.take(1)(0).size - 1 // -1 because one field is the value


    // Maps vector values and produces outgoing edges
    var tensorData: RDD[(Int, (Long, TTEdge))] = tensor.flatMap(v => {
      val dims = v.size - 1 // size of vector - 1 is dimensions
      val md = mdv.value
      val a = ListBuffer[(Int, (Long, TTEdge))]()
      for (i <- 0 until v.size - 1) { // - 1 because last index is the tensor value
        var ind = getRootIndex(dims, md) // The item which should emit a 0 in the kv pair
        ind = (ind + i) % dims // Now we emit  i and then the corresponding (val, TTEdge) pairs
        if (md != ind) {
          // Emit <i, TTEdge(rootIndexVal, nextTensorIndex)
          val edge = TTEdge(v(getRootIndex(dims, md)).toLong, v(nextTensorIndex(dims, ind)).toLong)
          a += Tuple2(i.toInt, (v(ind).toLong, edge))
        }
      }
      a.toList
    }) // Now we can filter out by key (array index), and put them into the graph array (gr)
    // However we still need to create the final RDD with pair <mode, val>
//    val modeData = tensor.map(v => {
//      (v(mode), v(v.size-1))
//    })
    for (i <- 0 until this.dims - 1) {
      this.gr(i) = tensorData.filter(f => f._1 == i).values
        .combineByKey( (e: TTEdge) => List(e),
          (l: List[TTEdge], e: TTEdge) => e :: l,
          (l: List[TTEdge], l2: List[TTEdge]) => l ::: l2)
    }
    this.imode = tensor.map( (v: Vector) => {
      Tuple2(v(mdv.value).toLong, v(v.size-1))
    })
  }

  def computeM1(mi: Array[IRowMatrix],
                SizeOfMatrix: Long,
                rank: Int): IRowMatrix = {

    // Even though it is verbose, strong typing helps sanity check during development
    val  joinedVecs: Array[RDD[(Long, (List[TTEdge], BDV[Double]))]] =
      new Array[RDD[(Long, (List[TTEdge], BDV[Double]))]](this.gr.length)


    // Join all of the vectors to their respective RDDs
    for (i <- 0 until this.gr.length) {
      joinedVecs(i) = this.gr(i).join(mi(i).rows)
    }

    // Propagate the Vectors down the network
    // First do the 0th item, then the 1st, until n-1 in a loop
    var last = joinedVecs(0).flatMap( { case (src: Long, (l: List[TTEdge], v: BDV[Double])) => {
      //For each edge in the list emit the KV pair <DEST, (EDGE, v)>
      // This let's us join with the next layer in the graph.
      val a = ListBuffer[(Long, VecInfo)]()
      l.foreach(edge => {
        a += Tuple2(edge.dst, VecInfo(edge.TreeNum, v))
      })
      a
    }}).combineByKey(List(_),
      (l: List[VecInfo], vi: VecInfo) => vi :: l,
      (l: List[VecInfo], l2: List[VecInfo]) => l ::: l2) // Put all of the vectors headed for the same edge together

    for (i <- 1 until this.gr.length) {
      var tmp = joinedVecs(i).join(last)
      // After joining with the vectors, multiply
      last = tmp.flatMap({ case (vertex:Long, ((outEdges: List[TTEdge], curVec: BDV[Double]), inVecs: List[VecInfo])) => {
        val l = ListBuffer[(Long, VecInfo)]()
        val a: mutable.HashMap[Long, BDV[Double]] = new mutable.HashMap[Long, BDV[Double]]()
        inVecs.foreach(vi => {
          a += Tuple2(vi.TreeNum, vi.vec :* curVec)
        })

        outEdges.foreach(edge => {
          if (a.contains(edge.TreeNum)) {
            // emit the dst and the new VecInfo
            l += Tuple2( edge.dst, VecInfo(edge.TreeNum, a(edge.TreeNum)))
          } else {
            println(s"For some reason we have a missing input treenum...$edge")
          }
        })
        l
      }}).combineByKey(List(_),
        (l: List[VecInfo], vi: VecInfo) => vi :: l,
        (l: List[VecInfo], l2: List[VecInfo]) => l ::: l2) // Put all of the vectors headed for the same edge together
    }

    var tmp = this.imode.join(last)
      .flatMap({
        case (vertex:Long, (v: Double, inVecs: List[VecInfo])) => {
          inVecs.map(f => Tuple2(vertex, f.vec :*= v))
        }
      })
//      .sortByKey() // does this improve performance? ==> It took over 50 seconds extra with this line.
      .reduceByKey(_ + _)

    new IRowMatrix(tmp)
  }

}
