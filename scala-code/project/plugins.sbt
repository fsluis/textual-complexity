resolvers += Resolver.url("sbt-plugin-releases-scalasbt", url("http://repo.scala-sbt.org/scalasbt/sbt-plugin-releases/"))

//https://github.com/sbt/sbt-assembly/
//addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.10.1")
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.10")

// http://stackoverflow.com/questions/10773319/sbt-doesnt-find-file-in-local-maven-repository-although-its-there
resolvers += "Local Maven" at Path.userHome.asFile.toURI.toURL + ".m2/repository"

// Next two are for Scalatra
addSbtPlugin("com.mojolly.scalate" % "xsbt-scalate-generator" % "0.4.2")

addSbtPlugin("org.scalatra.sbt" % "scalatra-sbt" % "0.3.2")

addSbtPlugin("org.bytedeco" % "sbt-javacpp" % "1.17")

// For dependencyTree command
addSbtPlugin("net.virtual-void" % "sbt-dependency-graph" % "0.9.2")
