
name := "spark-extkey"

organization := "ix"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.9.3"

scalacOptions += "-deprecation"

resolvers += "Akka Repo" at "http://repo.akka.io/repository"

// LOCAL
libraryDependencies ++= Seq(
)

// EXTERNAL
libraryDependencies ++= Seq(
    // Only available in test scope
    //"org.scalatest" % "scalatest_2.9.0" % "1.9.2" % "test",
    "org.scalatest" %% "scalatest" % "2.2.5" % "test",
    "de.sven-jacobs" % "loremipsum" % "1.0" % "test",
    // Only available at runtime
    "org.slf4j" % "slf4j-log4j12" % "1.7.5" % "runtime",
    "log4j" % "log4j" % "1.2.17" % "runtime",
    // Only available in compilation and test scopes
    //"org.apache.spark" % "spark-core_2.9.3" % "0.8.0-incubating" % "provided",
    "org.apache.spark" % "spark-core_2.11" % "2.4.3" % "provided",
    "org.apache.spark" % "spark-sql_2.11" % "2.4.3" % "provided",
    // And the normal ones, available in all scopes
    "org.slf4j" % "slf4j-api" % "1.6.6",
    // Apache configuration has an optional dependency on collections
    // http://stackoverflow.com/questions/28692844/why-is-maven-not-resolving-all-dependencies-for-commons-configuration
    "commons-collections" % "commons-collections" % "3.2.1" % "runtime"
)

// https://groups.google.com/forum/#!topic/simple-build-tool/seu72RRMd0k
parallelExecution in Test := false