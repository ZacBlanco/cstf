name := "CSTF"

version := "0.1"

scalaVersion := "2.10.4"

libraryDependencies ++= {
  val sparkVer = "1.5.2"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer,// % "provided" withSources(),
    "org.apache.spark" %% "spark-mllib" % sparkVer,// % "provided" withSources(),
    "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
    "org.scalactic" %% "scalactic" % "3.0.4",
    "org.scalatest" %% "scalatest" % "3.0.4" % "test"

  )
}