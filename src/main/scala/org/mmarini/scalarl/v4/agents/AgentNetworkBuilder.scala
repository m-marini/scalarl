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

package org.mmarini.scalarl.v4.agents

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
 *
 */
object AgentNetworkBuilder extends LazyLogging {

  /**
   * Returns the [[MultiLayerNetwork]] builder for an actor
   *
   * @param conf     the json configuration
   * @param noInputs the number of inputs
   * @param outputs  the outputs configuration
   */
  def fromJson(conf: ACursor)(noInputs: Int, outputs: Seq[Int]): ComputationGraph = {
    val seed = conf.get[Long]("seed").toOption
    val numHiddens = conf.get[List[Int]]("numHiddens").toTry.get
    val maxAbsGradient = conf.get[Double]("maxAbsGradients").toTry.get
    val maxAbsParams = conf.get[Double]("maxAbsParameters").toTry.get
    val dropOut = conf.get[Double]("dropOut").toTry.get
    val bias = conf.get[Double]("bias").toOption

    // Computes the number of inputs node for each layer
    val noHiddenInputs = if (numHiddens.isEmpty) {
      Seq()
    } else {
      noInputs +: numHiddens.init
    }

    val noOutputLayerInputs = if (numHiddens.isEmpty) {
      noInputs
    } else {
      numHiddens.last
    }

    // Creates the hidden layers
    val hiddenLayers = for {
      (outs, ins) <- numHiddens.zip(noHiddenInputs)
    } yield new DenseLayer.Builder().
      nIn(ins).
      nOut(outs).
      activation(Activation.TANH).
      dropOut(dropOut).
      build()

    val outputLayers = for {
      outs <- outputs
    } yield {
      val layer0 = new OutputLayer.Builder().
        nIn(noOutputLayerInputs).
        nOut(outs).
        lossFunction(LossFunction.MSE).
        activation(Activation.IDENTITY)
      val layer1 = bias.map(layer0.biasInit).getOrElse(layer0).build()
      layer1
    }

    // Computes num parameters
    val hiddenParms = hiddenLayers.map(l =>
      l.getNOut * (l.getNIn + 1)
    ).sum
    val outParms = outputLayers.map(l =>
      l.getNOut * (l.getNIn + 1)
    ).sum
    val noParms = outParms + hiddenParms

    val updater = UpdaterBuilder.fromJson(conf)(noParms.toInt)

    val annConf = seed.map(seed =>
      new NeuralNetConfiguration.Builder().seed(seed)
    ).getOrElse(new NeuralNetConfiguration.Builder()).
      weightInit(WeightInit.XAVIER).
      updater(updater).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      constrainAllParameters(new MinMaxNormConstraint(-maxAbsParams, maxAbsParams, 1)).
      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      gradientNormalizationThreshold(maxAbsGradient).
      graphBuilder().
      addInputs("L0")

    val withHidden = hiddenLayers.zipWithIndex.foldLeft(annConf) {
      case (builder, (layer, i)) => builder.addLayer(s"L${i + 1}", layer, s"L$i")
    }
    val inName = s"L${numHiddens.length}"
    val withOutput = outputLayers.zipWithIndex.foldLeft(withHidden) {
      case (builder, (layer, i)) => builder.addLayer(s"O$i", layer, inName)
    }
    val outNames = for {
      i <- 0 until outputLayers.length
    } yield s"O$i"
    val config = withOutput.setOutputs(outNames.toArray: _*).build()
    val net = new ComputationGraph(config)
    net.init()
    net
  }

  /**
   * Returns the [[MultiLayerNetwork]] builder for an actor
   *
   * @param file the file of loading model
   */
  def load(file: File): MultiLayerNetwork = {
    logger.info("Loading {} ...", file)
    ModelSerializer.restoreMultiLayerNetwork(file, true)
  }
}
