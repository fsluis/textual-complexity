//import AssemblyKeys._

// put this at the top of the file

name := "data"

version := "0.1-SNAPSHOT"

//scalaVersion := "2.10.5"

// Because of the interoperability with the hadoop sequence files... (@ data module)
scalacOptions += "-target:jvm-1.7"

javacOptions ++= Seq("-source", "1.7", "-target", "1.7")

// Resources... (can't make this global :S)
unmanagedResourceDirectories in Compile += {
  baseDirectory.value / "resources"
}

// LOCAL
libraryDependencies ++= Seq(
  "de.tudarmstadt.ukp.wikipedia" % "de.tudarmstadt.ukp.wikipedia.api" % "1.2.0-SNAPSHOT", // from local maven repo
  "de.tudarmstadt.ukp.wikipedia" % "de.tudarmstadt.ukp.wikipedia.parser" % "1.2.0-SNAPSHOT"
)

// EXTERNAL
libraryDependencies ++= Seq(
  "org.slf4j" % "slf4j-api" % "1.6.6",
  "org.slf4j" % "slf4j-log4j12" % "1.7.5" % "runtime",
  "log4j" % "log4j" % "1.2.17" % "runtime",
  "org.apache.hadoop" % "hadoop-client" % "2.7.3" % "provided",
  "org.apache.spark" % "spark-core_2.11" % "2.4.3" % "provided",
  "org.apache.spark" % "spark-sql_2.11" % "2.4.3" % "provided",
  "org.apache.spark" % "spark-mllib_2.11" % "2.4.3" % "provided",
  "org.apache.commons" % "commons-lang3" % "3.3.1",
  ("com.databricks" %% "spark-xml" % "0.9.0").excludeAll(ExclusionRule("org.glassfish.jaxb", "txw2"))
)

// https://mvnrepository.com/artifact/mysql/mysql-connector-java
libraryDependencies += "mysql" % "mysql-connector-java" % "8.0.19" //% "provided"

// https://mvnrepository.com/artifact/org.json4s/json4s-native_2.11
libraryDependencies += "org.json4s" %% "muster-codec-json4s" % "0.3.0"
// 3.5.3 is needed for parquet files, see https://stackoverflow.com/questions/65244646/java-lang-nosuchmethoderror-org-json4s-jsondsl-seq2jvaluelscala-collection-tr
libraryDependencies += "org.json4s" %% "json4s-native" % "3.5.3" // Matches Spark's dependency lower version
//libraryDependencies += "org.json4s" %% "json4s-native" % "4.0.6" // Matches Spark's upper version dependency

// https://mvnrepository.com/artifact/org.apache.commons/commons-math3
libraryDependencies += "org.apache.commons" % "commons-math3" % "3.6.1"



// This includes all, also 'provided' dependencies, during run
// http://stackoverflow.com/questions/18838944/how-to-add-provided-dependencies-back-to-run-test-tasks-classpath
run in Compile <<= Defaults.runTask(fullClasspath in Compile, mainClass in (Compile, run), runner in (Compile, run))

// Exclude Scala lib
assemblyOption in assembly ~= {
  _.copy(includeScala = false)
}

//logLevel in assembly := Level.Debug

// Don't run the tests here and now!
test in assembly := {}

assemblyMergeStrategy in assembly <<= (assemblyMergeStrategy in assembly) { (old) =>
  {
    case ".gitignore"     => MergeStrategy.discard
    case "META-INF/DEPENDENCIES.txt" => MergeStrategy.discard
    case "log4j.properties" => MergeStrategy.first
    case "pom.properties" => MergeStrategy.first
    case "module-info.class" => MergeStrategy.first
    // The following was added for deeplearning4java (except the old(x) line)
    // From https://stackoverflow.com/questions/61984594/assembly-scala-project-causes-deduplicate-errors
    //case PathList("META-INF", "services", xs@_*) => MergeStrategy.singleOrError
    //case "module-info.class" => MergeStrategy.singleOrError
    case PathList("org", "xmlpull", xs@_*) => MergeStrategy.discard
    case PathList("org", "nd4j", xs@_*) => MergeStrategy.first
    case PathList("org", "bytedeco", xs@_*) => MergeStrategy.first
    case PathList("org.bytedeco", xs@_*) => MergeStrategy.first
    case PathList("com", "google", "thirdparty", "publicsuffix", xs@_*) => MergeStrategy.first
    case PathList("com.google.thirdparty.publicsuffix", xs@_*) => MergeStrategy.first
    case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
    case PathList("META-INF", xs@_*) => MergeStrategy.first
    case "XmlPullParser.class" => MergeStrategy.discard
    case "Nd4jBase64.class" => MergeStrategy.discard
    case "XmlPullParserException.class" => MergeStrategy.discard
    //    case n if n.startsWith("rootdoc.txt") => MergeStrategy.discard
    //    case n if n.startsWith("readme.html") => MergeStrategy.discard
    //    case n if n.startsWith("readme.txt") => MergeStrategy.discard
    case n if n.startsWith("library.properties") => MergeStrategy.discard
    case n if n.startsWith("license.html") => MergeStrategy.discard
    case n if n.startsWith("about.html") => MergeStrategy.discard
    //    case _ => MergeStrategy.first
    case x => old(x)
  }
}
