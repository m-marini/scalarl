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

package org.mmarini.scalarl.v4.envs

import com.typesafe.scalalogging.LazyLogging
import monix.execution.Scheduler
import monix.execution.Scheduler.global
import org.mmarini.scalarl.v4.agents.AgentBuilder
import org.mmarini.scalarl.v4.reactive.Implicits._
import org.nd4j.linalg.factory.Nd4j._

import scala.concurrent.duration.DurationInt

/**
 *
 */
object Main extends LazyLogging {
  private implicit val scheduler: Scheduler = global

  def parseArgs(args: Array[String]): Map[String, String] = {
    val r = "--(.*)=(.*)".r
    val c = for {
      arg <- args
      s <- r.findAllMatchIn(arg)
      if s.groupCount == 2
    } yield {
      s.group(1) -> s.group(2)
    }
    c.toMap
  }

  /**
   *
   * @param args the line command arguments
   */
  def main(args: Array[String]) {
    create()

    val cfgParms = parseArgs(args)

    try {
      val file = cfgParms.get("conf").getOrElse("maze.yaml")
      val epoch = cfgParms.get("epoch").map(_.toInt).getOrElse(0)
      val kpiFile = cfgParms.get("kpiFile")
      val dumpFile = cfgParms.get("dumpFile")

      logger.info("File {} epoch {}", file, epoch)

      val jsonConf = Configuration.jsonFromFile(file)
      require(jsonConf.hcursor.get[String]("version").toTry.get == "4")

      val random = jsonConf.hcursor.get[Long]("seed").map(
        getRandomFactory.getNewRandomInstance
      ).getOrElse(
        getRandom
      )
      val env = EnvBuilder.fromJson(jsonConf.hcursor.downField("env"))(random)
      env.actionConfig
      val (agent, agentObs) = AgentBuilder.fromJson(jsonConf.hcursor.downField("agent"))(env.signalsSize, env.actionConfig)
      val session = SessionBuilder.fromJson(jsonConf.hcursor.downField("session"))(
        epoch,
        dumpFileParm = dumpFile,
        kpiFileParm = kpiFile,
        env = env,
        agent = agent,
        agentEvents = agentObs)

      session.lander().filterFinal().logInfo().subscribe()
      session.steps.monitorInfo().observable.sample(2 seconds).logInfo().subscribe()

      session.run(random)

      logger.info("Session completed.")
    } catch {
      case ex: Throwable =>
        logger.error(ex.getMessage, ex)
        throw ex
    }
  }
}
