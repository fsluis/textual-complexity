
name := "complexity-lm"

organization := "ix"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.10.5"

// mvn install:install-file -Dfile=/local/farm/src/kenlm-java/target/kenlm-0.0.6-SNAPSHOT.jar -DgroupId=com.github.jbaiter -DartifactId=kenlm -Dversion=0.0.6-SNAPSHOT -Dpackaging=jar
// mvn install:install-file -Dfile=/local/workspace/ix/ix-api/lib/kenlm-0.0.6-SNAPSHOT.jar -DgroupId=com.github.jbaiter -DartifactId=kenlm -Dversion=0.0.6-SNAPSHOT -Dpackaging=jar

resolvers += "Local Maven Repo" at Path.userHome.asFile.toURI.toURL + ".m2/repository"

// EXTERNAL
libraryDependencies ++= Seq(
  "com.google.guava" % "guava" % "18.0",
  "commons-logging" % "commons-logging" % "1.2",
  "org.xerial" % "sqlite-jdbc" % "3.8.11.2",
    "commons-collections" % "commons-collections" % "3.2.1" % "runtime",
    "org.scalatest" %% "scalatest" % "2.2.5" % "test",
  "org.apache.spark" % "spark-core_2.11" % "2.4.3" % "provided"
)
libraryDependencies += "net.liftweb" %% "lift-json" % "2.6.3"
libraryDependencies += "com.github.jbaiter" % "kenlm" % "0.0.6-SNAPSHOT"

//testOptions in Test += Tests.Argument("-oF")