// Copyright (c) 2019 Marco Marini, marco.marini@mmarini.org
//
// Licensed under the MIT License (MIT);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/MIT
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

package org.mmarini.scalarl.agents

import org.yaml.snakeyaml.Yaml
import org.opencv.dnn.Layer
import scala.collection.JavaConverters._
import java.util.Dictionary
import org.nd4j.linalg.api.ndarray.INDArray
import java.util.zip.ZipFile
import java.io.FileOutputStream
import java.util.zip.ZipOutputStream
import java.io.OutputStreamWriter
import java.util.zip.ZipEntry
import java.io.InputStreamReader
import org.mmarini.scalarl.envs.Configuration
import org.nd4j.linalg.factory.Nd4j
import java.io.File

object TraceModelSerializer {
  val EntryName = "net.yaml"

  def writeModel(net: TraceNetwork, file: String) {
    val metadata = createMetadata(net)
    val fout = new FileOutputStream(file)
    val zout = new ZipOutputStream(fout)
    val ze = new ZipEntry(EntryName)
    zout.putNextEntry(ze)
    val out = new OutputStreamWriter(zout, "UTF8")
    out.write(new Yaml().dump(metadata))
    out.flush()
    zout.closeEntry()
    zout.close()
    fout.close()
  }

  def createMetadata(net: TraceNetwork): java.util.Map[String, _ <: Any] = {
    val layersMeta = for {
      layer <- net.layers
    } yield layer match {
      case l: TraceDenseLayer =>
        createDenseMeta(l)
      case l: TraceTanhLayer =>
        Map("type" -> "TraceTanhLayer").asJava
    }

    Map(
      "loss" -> net.loss.name,
      "layers" -> layersMeta).asJava
  }

  def toMeta(tensor: INDArray): java.util.List[Double] = {
    val ravel = tensor.ravel()
    val meta = for {
      i <- 0L until ravel.size(1)
    } yield ravel.getDouble(i)
    meta.asJava
  }

  def createDenseMeta(net: TraceDenseLayer): java.util.Map[String, _ <: Any] = {
    Map(
      "type" -> "TraceDenseLayer",
      "inputs" -> net.weights.size(0),
      "outputs" -> net.weights.size(1),
      "gamma" -> net.gamma,
      "lambda" -> net.lambda,
      "weights" -> toMeta(net.weights),
      "bias" -> toMeta(net.bias),
      "learningRate" -> net.learningRate).asJava
  }

  def restoreTraceNetwork(file: File): TraceNetwork = {
    val zipFile = new ZipFile(file)
    val zipEntry = zipFile.getEntry(EntryName)
    val stream = zipFile.getInputStream(zipEntry)
    val reader = new InputStreamReader(stream, "UTF8")
    val conf = Configuration.fromReader(reader)
    reader.close()
    stream.close()
    zipFile.close()
    createNetwork(conf)
  }

  def createNetwork(conf: Configuration): TraceNetwork = {
    val layersConf = conf.getConfList("layers")
    val layers = for {
      layerConf <- layersConf
    } yield {
      val layerType = layerConf.getString("type")
      layerType match {
        case Some("TraceTanhLayer")  => TraceTanhLayer()
        case Some("TraceDenseLayer") => createDenseLayer(layerConf)
        case Some(unknown)           => throw new IllegalArgumentException(s"Unknown ${unknown} layer type")
        case _                       => throw new IllegalArgumentException("Missing layer type")
      }
    }
    require(conf.getString("loss").contains("MSE"), "Unknown loss")
    new TraceNetwork(
      layers = layers.toArray,
      loss = LossFunctions.MSE)
  }

  def createDenseLayer(conf: Configuration): TraceDenseLayer = {
    val inputs = conf.getLong("inputs").get
    val outputs = conf.getLong("outputs").get
    val weights = Nd4j.create(conf.getList[Double]("weights").toArray).reshape(inputs, outputs)
    val bias = Nd4j.create(conf.getList[Double]("bias").toArray).reshape(1, outputs)
    val gamma = conf.getDouble("gamma").get
    val lambda = conf.getDouble("lambda").get
    val learningRate = conf.getDouble("learningRate").get
    TraceDenseLayer(
      weights = weights,
      bias = bias,
      gamma = gamma,
      lambda = lambda,
      learningRate = learningRate)
  }
}
