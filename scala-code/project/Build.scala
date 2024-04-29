import sbt._

//http://alvinalexander.com/scala/using-github-projects-scala-library-dependencies-sbt-sbteclipse
object MyBuild extends Build {
  //lazy val data = Project("data", file("data")) dependsOn(pmmlSparkProject)
  //lazy val root = Project("root", file(".")) dependsOn(pmmlSpark)
  // http://blog.xebia.com/git-subproject-compile-time-dependencies-in-sbt/
  // Cloning into '/home/fsluis/.sbt/0.13/staging/5a4fdd17cfe1abb392e9/jpmml-spark'...
  // Cloning into '/Users/fsluis/.sbt/0.13/staging/5a4fdd17cfe1abb392e9/jpmml-spark'...
  //lazy val pmmlSpark = ProjectRef(uri("https://github.com/jpmml/jpmml-spark.git"), "pmml-spark")
}
