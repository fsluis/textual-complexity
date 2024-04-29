//import AssemblyKeys._ // put this at the top of the file

name := "integration"
// sbt +integration/publishLocal

organization := "ix"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.10.5"

// LOCAL
libraryDependencies ++= Seq(
)

// EXTERNAL
libraryDependencies ++= Seq(
    //"org.scalatest" % "scalatest_2.9.0" % "1.9.2" % "test",
    "org.scalatest" %% "scalatest" % "2.2.5" % "test",
    "de.sven-jacobs" % "loremipsum" % "1.0" % "test",
    "org.slf4j" % "slf4j-log4j12" % "1.7.5" % "runtime",
    "log4j" % "log4j" % "1.2.17" % "runtime",
    "org.apache.spark" % "spark-core_2.11" % "2.4.3" % "provided",
    "org.apache.spark" % "spark-sql_2.11" % "2.4.3" % "provided",
    //"org.daisy.bindings" % "jhyphen" % "0.1.5", //sadly, this needs to be double with complexity-hyphen project...
    "org.slf4j" % "slf4j-api" % "1.6.6",
    "org.apache.commons" % "commons-math3" % "3.2",
    "org.jpmml" % "pmml-evaluator" % "1.2.15",
    "org.jpmml" % "pmml-model" % "1.2.15" // need newer (1.2.x) version to work properly, but this clashes with Spark's version (until the new 2.x Spark comes out)
)

//scalacOptions in Compile ++= Seq("-Xprint-types", "-Xprint:typer")

// Assembly plugin
//assemblySettings

// your assembly settings here
// Exclude Scala lib
assemblyOption in assembly ~= { _.copy(includeScala = false) }

// Don't run the tests here and now!
test in assembly := {}

excludedJars in assembly <<= (fullClasspath in assembly) map { cp =>
  cp filter {_.data.getName.startsWith("spark-assembly")}
}