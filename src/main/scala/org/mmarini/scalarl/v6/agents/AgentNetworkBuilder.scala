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

package org.mmarini.scalarl.v6.agents

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

import java.io.File
import scala.util.{Failure, Success, Try}

/**
 *
 */
object AgentNetworkBuilder extends LazyLogging {

  /**
   * Returns the [[ComputationGraph]] builder for an actor
   *
   * @param conf     the json configuration
   * @param noInputs the number of inputs
   * @param outputs  the outputs configuration
   */
  def fromJson(conf: ACursor)(noInputs: Int, outputs: Seq[Int]): Try[ComputationGraph] = for {
    numHidden <- conf.get[Seq[Int]]("numHidden").toTry
    maxAbsGradient <- conf.get[Double]("maxAbsGradient").toTry
    maxAbsParams <- conf.get[Double]("maxAbsParameter").toTry
    dropOut <- conf.get[Double]("dropOut").toTry
    activation <- conf.get[String]("activation").toTry.flatMap {
      case "SOFTPLUS" => Success(Activation.SOFTPLUS)
      case "RELU" => Success(Activation.RELU)
      case "TANH" => Success(Activation.TANH)
      case "HARDTANH" => Success(Activation.HARDTANH)
      case "SIGMOID" => Success(Activation.SIGMOID)
      case "HARDSIGMOID" => Success(Activation.HARDSIGMOID)
      case act => Failure(new IllegalArgumentException(s"Wrong activation function $act"))
    }
    shortcuts = conf.get[Seq[Seq[Int]]]("shortcuts").toOption.getOrElse(Seq())
    shortcutsMap <- validateShortCut(shortcuts, numHidden.length)
  } yield {
    val seed = conf.get[Long]("seed").toOption

    // Computes the number of inputs for hidden layers
    val (noInputsByLayers, inputLayerNames) = inputLayersByLayer(noInputs, numHidden, shortcutsMap)
    val hiddenLayers = createHiddenLayers(noInputsByLayers.zip(numHidden), dropOut, activation)
    val outputLayers = createOutputLayers(outputs, noInputsByLayers.last)
    val noParams = numParams(hiddenLayers, outputLayers)
    val updater = UpdaterBuilder.fromJson(conf)(noParams.toInt)
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

    val withHidden = hiddenLayers.
      zip(inputLayerNames).
      zipWithIndex.
      foldLeft(annConf) {
        case (builder, ((layer, inputNames), i)) =>
          builder.addLayer(s"L${i + 1}", layer, inputNames.toArray: _*)
      }

    val inNames = inputLayerNames.last.toArray
    val withOutput = outputLayers.zipWithIndex.foldLeft(withHidden) {
      case (builder, (layer, i)) => builder.addLayer(s"O$i", layer, inNames: _*)
    }
    val outNames = for {
      i <- outputLayers.indices
    } yield s"O$i"
    val config = withOutput.setOutputs(outNames.toArray: _*).build()
    val net = new ComputationGraph(config)
    net.init()
    net
  }

  /**
   * Returns the validated shortcuts map
   *
   * @param shortcuts the shortcuts
   * @param noHidden  the number of hidden layers
   */
  def validateShortCut(shortcuts: Seq[Seq[Int]], noHidden: Int): Try[Map[Int, Seq[Int]]] = Try {
    // Validates the inputs
    // inputs, hidden, outputs
    for {
      (l, i) <- shortcuts.zipWithIndex
      from = l.head
      to = l(1)
    } {
      if (from < 0 || from > noHidden) {
        throw new IllegalArgumentException(s"Wrong input layer $from at shortcut $i")
      }
      if (to <= 0 || from > noHidden + 1) {
        throw new IllegalArgumentException(s"Wrong output layer $to at shortcut $i")
      }
      if (to <= from) {
        throw new IllegalArgumentException(s"Output layer $to must be forward than input layer $from at shortcut $i")
      }
    }
    // get the map of input layers by output layer
    val shortcutsMap = shortcuts.groupBy(_ (1)).map {
      case (out, list) => (out, list.map(_.head))
    }
    shortcutsMap
  }

  /**
   * Returns the hidden layers
   *
   * @param ioConf  list of number of inputs and outputs
   * @param dropOut the drop out parameter
   */
  def createHiddenLayers(ioConf: Seq[(Int, Int)], dropOut: Double, activation: Activation): Seq[DenseLayer] = {
    for {
      (ins, outs)
        <- ioConf
    } yield new DenseLayer.Builder().
      nIn(ins).
      nOut(outs).
      activation(activation).
      dropOut(dropOut).
      build()
  }

  /**
   * Returns the output layers
   *
   * @param outputs  the number of outputs nodes
   * @param noInputs the number of inputs
   */
  def createOutputLayers(outputs: Seq[Int], noInputs: Int): Seq[OutputLayer] = for {
    outs <- outputs
  } yield {
    val layer0 = new OutputLayer.Builder().
      nIn(noInputs).
      nOut(outs).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH)
    val layer1 = layer0.build()
    layer1
  }

  /**
   * Returns num parameters
   *
   * @param hidden  the hidden layers
   * @param outputs the output layers
   */
  def numParams(hidden: Seq[DenseLayer], outputs: Seq[OutputLayer]): Long = {
    val hiddenParams = hidden.map(l =>
      l.getNOut * (l.getNIn + 1)
    ).sum
    val outParams = outputs.map(l =>
      l.getNOut * (l.getNIn + 1)
    ).sum
    outParams + hiddenParams
  }

  /**
   * Returns the number of inputs per layer and the input names per layer
   *
   * @param noInputs     the number of inputs
   * @param hidden       the number of nodes for hidden layers
   * @param shortcutsMap the shortcuts
   */
  def inputLayersByLayer(noInputs: Int, hidden: Seq[Int], shortcutsMap: Map[Int, Seq[Int]]): (Seq[Int], Seq[Seq[String]]) = {
    // number of nodes per layer
    val noNodes = noInputs +: hidden
    // list of input layer for each layer
    val inputLayers = for {
      l <- 0 to hidden.length
    } yield {
      (l +: shortcutsMap.getOrElse(l + 1, Seq())).distinct
    }
    // Number of inputs for each layer
    val noLayerInputs = inputLayers.map(l => {
      l.map(noNodes).sum
    })
    // list of input layer name per layer
    val inputLayerNames = inputLayers.map(
      _.map(l => s"L$l")
    )
    (noLayerInputs, inputLayerNames)
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
