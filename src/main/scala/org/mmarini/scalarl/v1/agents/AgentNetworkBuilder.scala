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

package org.mmarini.scalarl.v1.agents

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v1.envs.AgentNetworkBuilder.DefaultSeed
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
 *
 */
object AgentNetworkBuilder extends LazyLogging {

  /**
   * Returns the [[MultiLayerNetwork]] builder for an actor
   *
   * @param conf      the json configuration
   * @param noInputs  the number of inputs
   * @param noOutputs the number of outputs
   */
  def fromJson(conf: ACursor)(noInputs: Int, noOutputs: Int): MultiLayerNetwork = conf.get[String]("modelFile").toOption.
    map(load).getOrElse {
    create(conf)(noInputs, noOutputs)
  }

  /**
   * Returns the [[MultiLayerNetwork]] builder for an actor
   *
   * @param conf      the json configuration
   * @param noInputs  the number of inputs
   * @param noOutputs the number of outputs
   */
  def create(conf: ACursor)(noInputs: Int, noOutputs: Int): MultiLayerNetwork = {
    val seed = conf.get[Long]("seed").getOrElse(DefaultSeed)
    val numHiddens = conf.get[List[Int]]("numHiddens").right.get
    val maxAbsGradient = conf.get[Double]("maxAbsGradients").right.get
    val maxAbsParams = conf.get[Double]("maxAbsParameters").right.get
    val dropOut = conf.get[Double]("dropOut").right.get
    val bias = conf.get[Double]("bias").toOption
    // Computes the number of nodes of initial layers
    val initialNodes = noInputs +: numHiddens
    // Creates the hidden layers
    val hiddenLayers = for {
      (ins, outs) <- initialNodes.init.zip(initialNodes.tail)
    } yield new DenseLayer.Builder().
      nIn(ins).
      nOut(outs).
      activation(Activation.TANH).
      dropOut(dropOut).
      build()
    val outLayerBuilder0 = new OutputLayer.Builder().
      nIn(initialNodes.last).
      nOut(noOutputs).
      lossFunction(LossFunction.MSE).
      activation(Activation.TANH)
    val outLayerBuilder = bias.map(outLayerBuilder0.biasInit).getOrElse(outLayerBuilder0)
    val outLayer = outLayerBuilder.build()

    // Computes num parameters per layer
    val hiddenParms = for {
      (ins, outs) <- initialNodes.init.zip(initialNodes.tail)
    } yield
      outs * (ins + 1)
    val outParms = (initialNodes.last + 1) * noOutputs
    val noParms = outParms + hiddenParms.sum

    val updater = UpdaterBuilder.fromJson(conf)(noParms)

    val annConf = new NeuralNetConfiguration.Builder().
      seed(seed).
      weightInit(WeightInit.XAVIER).
      updater(updater).
      optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).
      constrainAllParameters(new MinMaxNormConstraint(-maxAbsParams, maxAbsParams, 1)).
      gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).
      gradientNormalizationThreshold(maxAbsGradient).
      list(hiddenLayers :+ outLayer: _*).
      build()

    val net = new MultiLayerNetwork(annConf)
    net.init()
    net
  }

  /**
   * Returns the [[MultiLayerNetwork]] builder for an actor
   *
   * @param path the path of loading model
   */
  def load(path: String): MultiLayerNetwork = {
    val file = new File(path, "actor.zip")
    logger.info("Loading {} ...", file)
    ModelSerializer.restoreMultiLayerNetwork(file, true)
  }
}
