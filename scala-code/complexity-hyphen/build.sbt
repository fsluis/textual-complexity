
name := "complexity-hyphen"

organization := "ix"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.10.5"

resolvers += "Local Maven Repo" at Path.userHome.asFile.toURI.toURL + ".m2/repository"

// EXTERNAL
libraryDependencies ++= Seq(
    //"org.scalatest" % "scalatest_2.9.0" % "1.9.2" % "test",
    "org.scalatest" % "scalatest_2.10" % "2.2.5" % "test",
    "org.slf4j" % "slf4j-log4j12" % "1.7.5" % "runtime",
    "log4j" % "log4j" % "1.2.17" % "runtime",
    "org.slf4j" % "slf4j-api" % "1.6.6",
    "commons-configuration" % "commons-configuration" % "1.10",
    "commons-collections" % "commons-collections" % "3.2.1" % "runtime",
    "org.daisy.bindings" % "jhyphen" % "1.0.2",
    "de.sven-jacobs" % "loremipsum" % "1.0" % "test"
)

//unmanagedJars in Compile := (file("lib/") ** "*.jar").classpath