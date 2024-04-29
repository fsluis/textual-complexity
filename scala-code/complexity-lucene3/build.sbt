
name := "complexity-lucene3"

organization := "ix"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.10.5"

// org.apache.lucene3 is in local maven (manually hashed) - see shared subfolder, run maven install there
resolvers += Resolver.mavenLocal

// LOCAL
libraryDependencies ++= Seq(
    "org.apache.lucene3" % "lucene-core" % "3.6.2",
    "org.apache.lucene3" % "lucene-analyzers" % "3.6.2"
)

// EXTERNAL
libraryDependencies ++= Seq(
    //"org.scalatest" % "scalatest_2.9.0" % "1.9.2" % "test",
    "org.scalatest" % "scalatest_2.10" % "2.2.5" % "test",
    "de.sven-jacobs" % "loremipsum" % "1.0" % "test",
    "org.slf4j" % "slf4j-log4j12" % "1.7.5" % "runtime",
    "log4j" % "log4j" % "1.2.17" % "runtime",
    "org.slf4j" % "slf4j-api" % "1.6.6",
    "commons-configuration" % "commons-configuration" % "1.10",
    "net.sf.trove4j" % "trove4j" % "3.0.3",
     "commons-collections" % "commons-collections" % "3.2.1" % "runtime",
     "net.sourceforge.collections" % "collections-generic" % "4.01"
)

