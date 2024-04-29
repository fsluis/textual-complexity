
name := "complexity-stanford"

organization := "ix"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.10.5"

//resolvers += "Local Maven Repo" at Path.userHome.asFile.toURI.toURL + ".m2/repository"

// LOCAL JARS
//unmanagedJars in Compile += file("lib/mtj.jar")
libraryDependencies ++= Seq(
    "com.googlecode.matrix-toolkits-java" % "mtj" % "1.0.5-SNAPSHOT"
)

// EXTERNAL
libraryDependencies ++= Seq(
    //Stanford CoreNLP 3.5.2 was original. 3.7.0 and higher includes lucene4, can't use!
    ("edu.stanford.nlp" % "stanford-corenlp" % "3.9.2"), //.excludeAll(ExclusionRule("com.google.protobuf", "protobuf-java")),
    "commons-collections" % "commons-collections" % "3.2.1" % "runtime",
    "org.scalatest" %% "scalatest" % "2.2.5" % "test"
)

