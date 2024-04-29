
name := "common-scala"

organization := "ix"

version := "0.2-SNAPSHOT"

// LOCAL
libraryDependencies ++= Seq(
  "com.googlecode.matrix-toolkits-java" % "mtj" % "1.0.5-SNAPSHOT"
)

// EXTERNAL
libraryDependencies ++= Seq(
    "commons-jxpath" % "commons-jxpath" % "1.3",
    "commons-configuration" % "commons-configuration" % "1.10",
    // Configuration has an optional dependency on collections
    // http://stackoverflow.com/questions/28692844/why-is-maven-not-resolving-all-dependencies-for-commons-configuration
    "commons-collections" % "commons-collections" % "3.2.1" % "runtime",
    "org.slf4j" % "slf4j-api" % "1.6.6",
    //from here on, the connectors/interfaces are optional. Import furtheron in the dependency tree when needed!
  "org.apache.spark" % "spark-core_2.11" % "2.4.3" % "provided",
  "org.apache.spark" % "spark-sql_2.11" % "2.4.3" % "provided",
  // later versions remove a method needed by rogue...
  // https://jira.mongodb.org/browse/JAVA-1543
  //"com.googlecode.matrix-toolkits-java" % "mtj" % "1.0.3",
  "log4j" % "log4j" % "1.2.17" % "runtime",
  "org.slf4j" % "slf4j-log4j12" % "1.7.5" % "runtime",
  "org.slf4j" % "slf4j-api" % "1.6.6"
)

libraryDependencies += "mysql" % "mysql-connector-java" % "5.1.18" % "provided"
libraryDependencies += "org.postgresql" % "postgresql" % "9.4.1212" % "provided"
// https://mvnrepository.com/artifact/org.apache.commons/commons-dbcp2
libraryDependencies += "org.apache.commons" % "commons-dbcp2" % "2.2.0"
