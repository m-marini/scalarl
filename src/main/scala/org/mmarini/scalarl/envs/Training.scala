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

package org.mmarini.scalarl.envs

import java.io.File

import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.FileUtils
import org.mmarini.scalarl.agents.AgentNetworkBuilder
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import com.typesafe.scalalogging.LazyLogging

import io.circe.ACursor

object Training extends LazyLogging {
  private val ClearScreen = "\033[2J\033[H"
  private val NumFeatures = 21
  private val NumLabels = 15
  private val BatchSize = 150
  private val MonitorCounter = 100
  private val TrainRatio = 0.65
  private val QOffset = 21L
  private val ActionsOffset = 36L
  private val RewardOffset = 51L
  private val EndUpOffset = 52L

  /**
   * Returns the observable of samples dataset after processing for q values
   */
  private def loadData(conf: ACursor) = {
    val samplesFile = conf.get[String]("samples").right.get
    val batchSize = conf.get[Int]("batchSize").toOption.getOrElse(BatchSize)
    logger.info("Loading {} ...", samplesFile)
    FileUtils.readINDArray(new File(samplesFile)).map(toDataset)
  }

  /**
   *
   */
  private def processLanderData(x: INDArray): INDArray = {
    val y = flipud(x)
    val Array(n, m) = y.shape()
    val first = (0 until n.toInt).find(y.getInt(_, EndUpOffset.toInt) != 0).get
    val y1 = y.get(NDArrayIndex.interval(first, n - 1))
    val n1 = y1.size(0)
    val gamma = 0.999
    var ret = 0.0

    for {
      i <- 0 until n1.toInt
    } {
      val row = y1.getRow(i)
      val q = row.get(NDArrayIndex.interval(NumFeatures, NumFeatures + NumLabels))
      val actions = row.get(NDArrayIndex.interval(ActionsOffset, ActionsOffset + NumLabels))
      val endUp = row.getInt(EndUpOffset.toInt)
      val reward = row.getDouble(RewardOffset)
      if (endUp != 0) {
        ret = 0.0
      }
      ret = ret * gamma + reward
      for {
        j <- 0 until NumLabels
        action = actions.getInt(j)
        if action != 0
      } {
        y1.putScalar(Array(i, j + QOffset), ret)
      }
    }

    val y2 = flipud(y1.get(NDArrayIndex.all(), NDArrayIndex.interval(0, NumFeatures + NumLabels)))
    y2
  }

  /** Returns the samples dataset by processing for q values */
  private def toDataset(x: INDArray): DataSet = {
    val y = processLanderData(x)
    val features = y.get(NDArrayIndex.all(), NDArrayIndex.interval(0, NumFeatures))
    val labels = y.get(NDArrayIndex.all(), NDArrayIndex.interval(NumFeatures, NumFeatures + NumLabels))
    val result = new DataSet(features, labels)
    result
  }

  /**
   *
   */
  private def toTestAndTrain(x: DataSet): (DataSet, DataSet) = {
    x.shuffle()
    val testAndTrain = x.splitTestAndTrain(TrainRatio)

    val trainingData = testAndTrain.getTrain()
    val testData = testAndTrain.getTest()
    (trainingData, testData)
  }

  /**
   *
   */
  def main(args: Array[String]) {
    val file = if (args.isEmpty) "maze.yaml" else args(0)
    logger.info("File {}", file)

    val jsonConf = Configuration.jsonFromFile(file)

    val netConf = jsonConf.hcursor.downField("network")
    val net = AgentNetworkBuilder(netConf).build()

    val trainingConf = jsonConf.hcursor.downField("training")
    loadData(trainingConf).map(toTestAndTrain).subscribe(_ match {
      case (trainingData, testData) =>
        try {
          net.setListeners(new ScoreIterationListener(MonitorCounter))
          val epochs = trainingConf.get[Int]("epochs").right.get

          logger.info("Fitting ...")
          for {
            i <- Range(1, epochs)
          } {
            net.fit(trainingData)
          }
          logger.info("Fit.")

          val builder = netConf.get[String]("modelFile").toOption.
            foreach(file => {
              ModelSerializer.writeModel(net, new File(file), false)
              logger.info(s"Written ${file}")
            })

          val eval = new RegressionEvaluation(NumLabels)
          val output = net.output(testData.getFeatures())
          eval.eval(testData.getLabels(), output);
          logger.info(eval.stats())
        } catch {
          case ex: Throwable => logger.error(ex.getMessage, ex)
        }
    })
  }

  /**
   *
   */
  private def flipud(x: INDArray): INDArray = {
    val Array(n, m) = x.shape()
    val result = Nd4j.createUninitialized(n, m)
    for { i <- 0L until n } {
      val row = x.get(NDArrayIndex.point(i))
      result.get(NDArrayIndex.point(n - i - 1)).assign(row)
    }
    result
  }
}
