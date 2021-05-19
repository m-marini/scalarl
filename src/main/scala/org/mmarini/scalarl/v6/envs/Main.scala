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

package org.mmarini.scalarl.v6.envs

import com.typesafe.scalalogging.LazyLogging
import monix.execution.Scheduler
import monix.execution.Scheduler.global
import org.mmarini.scalarl.v6.Configuration
import org.mmarini.scalarl.v6.agents.AgentBuilder
import org.mmarini.scalarl.v6.reactive.Implicits._
import org.nd4j.linalg.factory.Nd4j._

import java.time.Clock
import scala.concurrent.duration.DurationInt

/**
 *
 */
object Main extends LazyLogging {
  private implicit val scheduler: Scheduler = global

  /**
   *
   * @param args the line command arguments
   */
  def main(args: Array[String]): Unit = {
    create()

    val cfgParams = parseArgs(args)

    try {
      val file = cfgParams.getOrElse("conf", "maze.yaml")
      val epoch = cfgParams.get("epoch").map(_.toInt).getOrElse(0)
      val kpiFile = cfgParams.get("kpiFile")
      val dumpFile = cfgParams.get("dumpFile")
      val traceFile = cfgParams.get("traceFile")

      logger.info("File {} epoch {}", file, epoch)

      val jsonConf = Configuration.jsonFromFile(file)
      require(jsonConf.hcursor.get[String]("version").toTry.get == "6")

      val random = jsonConf.hcursor.get[Long]("seed").map(
        getRandomFactory.getNewRandomInstance
      ).getOrElse(
        getRandom
      )
      val env = EnvBuilder.fromJson(jsonConf.hcursor.downField("env"))(random).get

      val agent = AgentBuilder.fromJson(jsonConf.hcursor.downField("agent"))(env.signalsSize, env.actionDimensions).get
      val session = SessionBuilder.fromJson(jsonConf.hcursor.downField("session"))(
        epoch,
        dumpFileParam = dumpFile,
        kpiFileParam = kpiFile,
        traceFileParam = traceFile,
        env = env,
        agent = agent,
        agentEvents = agent.conf.agentObserver).get

      session.lander().filterFinal().logInfo().subscribe()
      session.steps.monitorInfo().observable.sample(2 seconds).logInfo().subscribe()

      val start = Clock.systemDefaultZone().instant()
      session.run(random)

      val end = Clock.systemDefaultZone().instant()
      val elapsed = java.time.Duration.between(start, end)

      dumpFile.foreach(filename => {
        logger.info("Dump file {}", filename)
      })

      kpiFile.foreach(filename => {
        logger.info("Kpi file {}", filename)
      })

      logger.info("Session completed in {}.", formatDuration(elapsed))
    } catch {
      case ex: Throwable =>
        logger.error(ex.getMessage, ex)
        throw ex
    }
  }

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

  def formatDuration(duration: java.time.Duration): String = {
    val ms = duration.toMillis
    val days = ms / (24 * 60 * 60 * 1000)
    val ms1 = ms - days * (24 * 60 * 60 * 1000)
    val hours = ms1 / (60 * 60 * 1000)
    val ms2 = ms1 - hours * (60 * 60 * 1000)
    val min = ms2 / (60 * 1000)
    val secs = (ms2 - min * (60 * 1000)).toDouble / 1000.0

    val r = StringBuilder.newBuilder
    if (days != 0) {
      r.append(f"""$days%d days $hours%dh $min%d' $secs%.3f"""")
    } else if (hours != 0) {
      r.append(f"""$hours%dh $min%d' $secs%.3f"""")
    } else if (min != 0) {
      r.append(f"""$min%d' $secs%.3f"""")
    } else {
      r.append(f"""$secs%.3f"""")
    }
    r.toString
  }
}
