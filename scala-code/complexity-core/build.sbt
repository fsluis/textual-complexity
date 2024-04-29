
name := "complexity-core"

organization := "ix"

version := "0.1-SNAPSHOT"

// LOCAL
libraryDependencies ++= Seq(
    "com.googlecode.matrix-toolkits-java" % "mtj" % "1.0.5-SNAPSHOT"
)

// EXTERNAL
libraryDependencies ++= Seq(
    "com.google.guava" % "guava" % "r07",
    "org.scalatest" % "scalatest_2.10" % "2.2.5" % "test",
    "org.slf4j" % "slf4j-log4j12" % "1.7.5" % "runtime",
    "log4j" % "log4j" % "1.2.17" % "runtime",
    "org.scalatest" % "scalatest_2.10" % "2.2.5" % "test",
    "org.slf4j" % "slf4j-api" % "1.6.6",
    //"org.scala-lang" % "scala-actors" % "2.10.5",
    "org.apache.commons" % "commons-math3" % "3.2",
    "org.scala-lang.modules" %% "scala-java8-compat" % "0.8.0"
)

testOptions in Test += Tests.Argument("-oS")
