package org.paramath.CSTF.utils

import breeze.linalg.split
import org.apache.spark.executor.{ShuffleReadMetrics, ShuffleWriteMetrics}
import org.apache.spark.scheduler._
import org.paramath.CSTF.utils.TaskInfoRecorderListener.{TaskAccumulablesInfo, TaskVals}

import scala.collection.mutable.ListBuffer

class TaskInfoRecorderListener(gatherAccumulables: Boolean = false) extends SparkListener {

  val taskMetricsData: ListBuffer[TaskVals] = new ListBuffer()
  val accumulablesMetricsData: ListBuffer[TaskAccumulablesInfo] = ListBuffer.empty[TaskAccumulablesInfo]
  val StageIdtoJobId: collection.mutable.HashMap[Int, Int] = collection.mutable.HashMap.empty[Int, Int]

  def encodeTaskLocality(taskLocality: TaskLocality.TaskLocality): Int = {
    taskLocality match {
      case TaskLocality.PROCESS_LOCAL => 0
      case TaskLocality.NODE_LOCAL => 1
      case TaskLocality.RACK_LOCAL => 2
      case TaskLocality.NO_PREF => 3
      case TaskLocality.ANY => 4
    }
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    jobStart.stageIds.foreach(stageId => StageIdtoJobId += (stageId -> jobStart.jobId))
  }

  /**
    * This methods fires at the end of the Task and collects metrics flattened into the taskMetricsData ListBuffer
    * Note all times are in ms, cpu time and shufflewrite are originally in nanosec, thus in the code are divided by 1e6
    */
  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    val taskInfo = taskEnd.taskInfo
    val taskMetrics = taskEnd.taskMetrics
    val gettingResultTime = {
      if (taskInfo.gettingResultTime == 0L) 0L
      else taskInfo.finishTime - taskInfo.gettingResultTime
    }
    val duration = taskInfo.finishTime - taskInfo.launchTime
    val jobId = StageIdtoJobId(taskEnd.stageId)
    val currentTask = TaskVals(jobId, taskEnd.stageId, taskInfo.taskId, taskInfo.launchTime,
      taskInfo.finishTime, duration,
      math.max(0L, duration - taskMetrics.executorRunTime - taskMetrics.executorDeserializeTime -
        taskMetrics.resultSerializationTime - gettingResultTime),
      taskInfo.executorId, taskInfo.host, encodeTaskLocality(taskInfo.taskLocality),
      taskInfo.speculative, gettingResultTime, taskInfo.successful,
      taskMetrics.executorRunTime, taskMetrics.executorDeserializeTime,
      taskMetrics.resultSerializationTime, taskMetrics.jvmGCTime, taskMetrics.resultSize,
      taskMetrics.diskBytesSpilled, taskMetrics.memoryBytesSpilled,
      taskMetrics.shuffleReadMetrics,
      taskMetrics.shuffleWriteMetrics)
    taskMetricsData += currentTask

    /** Collect data from accumulators (includes task metrics and SQL metrics)
      * as this can be a lot of data, only gather data if gatherAccumulables is true
      * note the additional filters to keep only numerical values to make this code simpler
      * note and todo: gatherAccumulables for TaskMetrics implementation currently works only for spark 2.1.x,
      * this feature is broken on 2.2.1 as a consequence of [SPARK PR 17596](https://github.com/apache/spark/pull/17596)
      */
    if (gatherAccumulables) {
      taskInfo.accumulables.foreach(acc => try {
        val value = acc.value.asInstanceOf[Long]
        val name = acc.name
        val currentAccumulablesInfo = TaskAccumulablesInfo(jobId, taskEnd.stageId, taskInfo.taskId,
          taskInfo.launchTime, taskInfo.finishTime, acc.id, name, value)
        accumulablesMetricsData += currentAccumulablesInfo
      }
      catch {
        case ex: ClassCastException => None
      }
      )
    }

  }

  def gatherTaskVals(key: String): String  = {
    if (taskMetricsData.isEmpty) {
      "-1"
    } else {
      try {
        var t1 = taskMetricsData.toArray.map(t => TaskInfoRecorderListener.taskValsString(t))
        var t2 = t1.flatMap(f => f.split('\n'))
        var t3 = t2.filter(s => s.contains(key))
        if (t3.length > 0) {
          var t4 = t3.map( (l: String) => {
            val x: Array[String] = l.split('|')
            x(2).replace(" ", "").stripLineEnd
          })
          var t5 = t4.map(x => x.toLong)
          t5.reduce((v1, v2) => v1+v2).toString
        } else {
          "-1"
        }
      } catch {
        case e: Exception => "-1"
      }

    }
  }
}
object TaskInfoRecorderListener {


  case class TaskAccumulablesInfo(jobId: Int,
                                  stageId: Int,
                                  taskId: Long,
                                  submissionTime: Long,
                                  finishTime: Long,
                                  accId: Long,
                                  name: String,
                                  value: Long)
  case class TaskVals(jobId: Int,
                      stageId: Int,
                      index: Long,
                      launchTime: Long,
                      finishTime: Long,
                      duration: Long,
                      schedulerDelay: Long,
                      executorId: String,
                      host: String,
                      taskLocality: Int,
                      speculative: Boolean,
                      gettingResultTime: Long,
                      successful: Boolean,
                      executorRunTime: Long,
                      executorDeserializeTime: Long,
                      resultSerializationTime: Long,
                      jvmGCTime: Long,
                      resultSize: Long,
                      diskBytesSpilled: Long,
                      memoryBytesSpilled: Long,
                      shuffleReadMetrics: Option[ShuffleReadMetrics],
                      shuffleWriteMetrics: Option[ShuffleWriteMetrics])

  def taskValsString(t: TaskVals): String = {
    val id = "Executor:" +  t.executorId.toString + " jobID: " + t.jobId .toString + " stageID " + t.stageId.toString
    def p(k: String, v:String): String = {
      val s = s"$id | $k | $v\n"
      s
    }
    var s = ""

    s += p("launchTime", t.launchTime.toString)
    s += p("finishTime", t.finishTime.toString)
    s += p("duration", t.duration.toString)
    s += p("schedulerDelay", t.schedulerDelay.toString)
    s += p("taskLocality", t.taskLocality.toString)
    s += p("speculative", t.speculative.toString)
    s += p("gettingResultTime", t.gettingResultTime.toString)
    s += p("successful", t.successful.toString)
    s += p("executorRunTime", t.executorRunTime.toString)
    s += p("executorDeserializeTime", t.executorDeserializeTime.toString)
    s += p("resultSerializationTime", t.resultSerializationTime.toString)
    s += p("jvmGCTime", t.jvmGCTime.toString)
    s += p("resultSize", t.resultSize.toString)
    s += p("diskBytesSpilled", t.diskBytesSpilled.toString)
    s += p("memoryBytesSpilled", t.memoryBytesSpilled.toString)
    if (t.shuffleReadMetrics.getOrElse(0) != 0) {
      val m = t.shuffleReadMetrics.get
      s += p("fetchWaitTime", m.fetchWaitTime.toString)
      s += p("localBlocksFetched", m.localBlocksFetched.toString)
      s += p("localBytesRead", m.localBytesRead.toString)
      s += p("recordsRead", m.recordsRead.toString)
      s += p("remoteBlocksFetched", m.remoteBlocksFetched.toString)
      s += p("remoteBytesRead", m.remoteBytesRead.toString)
      s += p("totalBlocksFetched", m.totalBlocksFetched.toString)
      s += p("totalBytesRead", m.totalBytesRead.toString)
    }
    if (t.shuffleWriteMetrics.getOrElse(0) != 0) {
      val m = t.shuffleWriteMetrics.get
      s += p("shuffleBytesWritten", m.shuffleBytesWritten.toString)
      s += p("shuffleRecordsWritten", m.shuffleRecordsWritten.toString)
      s += p("shuffleWriteTime", m.shuffleWriteTime.toString)
    }

    s
  }
}