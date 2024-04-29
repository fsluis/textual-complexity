//import AssemblyKeys._ // put this at the top of the file

name := "spark-test"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.9.3"

// http://www.scala-sbt.org/0.12.2/docs/Howto/logging.html
traceLevel in Test := 5

testOptions in Test += Tests.Argument("-w ix.util.spark.tests")

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
