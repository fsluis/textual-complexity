
name := "common"
// sbt +common_scala/publishLocal

organization := "ix"

version := "0.1-SNAPSHOT"

// Enables publishing to maven repo
publishMavenStyle := true

// Do not append Scala versions to the generated artifacts
crossPaths := false

// This forbids including Scala related libraries into the dependency
autoScalaLibrary := false

// LOCAL
//mvn install:install-file -Dfile=lib/mtj.jar -DgroupId=com.googlecode.matrix-toolkits-java -DartifactId=mtj -Dversion=1.0.5-SNAPSHOT -Dpackaging=jar
libraryDependencies ++= Seq(
  "com.googlecode.matrix-toolkits-java" % "mtj" % "1.0.5-SNAPSHOT"
)

// EXTERNAL
libraryDependencies ++= Seq(
  "com.google.guava" % "guava" % "18.0", //% "provided", // added the provided tag to prevent future clashes with dl4j...
  "commons-logging" % "commons-logging" % "1.2",
  "org.apache.commons" % "commons-math3" % "3.5",
//  "com.googlecode.matrix-toolkits-java" % "mtj" % "1.0.3",
  // I removed RootedTree.java because it was the only dependency on jgrapht, which had a collision with the wikipedia api of the data module
  //"org.jgrapht" % "jgrapht-core" % "0.9.1",
  "mysql" % "mysql-connector-java" % "8.0.19" % "provided"
)

// http://www.scala-sbt.org/1.0/docs/Howto-Scaladoc.html
// https://stackoverflow.com/questions/26049329/javadoc-in-jdk-8-invalid-self-closing-element-not-allowed
javacOptions in (Compile,doc) += "-Xdoclint:none"