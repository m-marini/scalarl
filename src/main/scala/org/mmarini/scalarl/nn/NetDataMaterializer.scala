package org.mmarini.scalarl.nn

import java.io.FileWriter
import java.io.Writer
import cats.syntax.either._
import io.circe.yaml._
import io.circe.yaml.syntax._
import io.circe.Json

object NetDataMaterializer {
  def toJson(data: NetworkData): Json = {
    val x = data.toArray.map {
      case (key, value) =>
        val a = value.toDoubleVector().flatMap(Json.fromDouble)
        val aj = Json.arr(a: _*)
        (key, aj)
    };
    Json.obj(x: _*)
  }

  def write(writer: Writer, data: NetworkData) {
    writer.write(toJson(data).asYaml.spaces2)
  }

  def write(file: String, data: NetworkData) {
    val fw = new FileWriter(file)
    write(fw, data)
    fw.close()
  }
  
}