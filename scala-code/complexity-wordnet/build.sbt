
name := "complexity_wordnet"

organization := "ix"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.10.5"

resolvers += "Local Maven Repo" at Path.userHome.asFile.toURI.toURL + ".m2/repository"

// LOCAL
//    "hyphen" % "hyphen" % "0.1"
libraryDependencies ++= Seq(
)

// EXTERNAL
libraryDependencies ++= Seq(
    "edu.mit" % "jwi" % "2.2.3"
)

//unmanagedJars in Compile := (file("lib/") ** "*.jar").classpath