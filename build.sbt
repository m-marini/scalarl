organization := "org.mmarini"

name := "scalarl"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.12.8"

libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.2"
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3"
libraryDependencies += "io.monix" %% "monix" % "3.1.0"
libraryDependencies += "io.circe" %% "circe-yaml" % "0.9.0"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta3"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta3"
libraryDependencies += "org.datavec" % "datavec-api" % "1.0.0-beta3"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % Test
libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.14.0" % Test

publishArtifact in(Compile, packageDoc) := false

lazy val root = (project in file("."))
  .enablePlugins(JavaAppPackaging)
  .settings(
    name := "Maze"
  )
