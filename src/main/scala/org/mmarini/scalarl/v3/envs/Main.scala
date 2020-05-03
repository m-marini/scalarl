package org.mmarini.scalarl.v3.envs

import com.typesafe.scalalogging.LazyLogging
import monix.eval.Task
import monix.execution.Scheduler
import monix.execution.Scheduler.global
import org.mmarini.scalarl.v3.agents.{ActorCriticAgent, AgentBuilder, PolicyActor}
import org.mmarini.scalarl.v3.reactive.WrapperBuilder._
import org.mmarini.scalarl.v3.{Feedback, Step, Utils}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms

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
  def main(args: Array[String]) {
    create()

    try {
      val file = if (args.isEmpty) "maze.yaml" else args(0)
      val epoch = if (args.length >= 2) args(1).toInt else 0
      logger.info("File {} epoch {}", file, epoch)

      val jsonConf = Configuration.jsonFromFile(file)
      require(jsonConf.hcursor.get[String]("version").toTry.get == "3")

      val random = jsonConf.hcursor.get[Long]("seed").map(
        getRandomFactory.getNewRandomInstance
      ).getOrElse(
        getRandom
      )
      val env = EnvBuilder.fromJson(jsonConf.hcursor.downField("env"))(random)
      val agent = AgentBuilder.fromJson(jsonConf.hcursor.downField("agent"))(env.signalsSize, env.actionConfig)
      val session = SessionBuilder.fromJson(jsonConf.hcursor.downField("session"))(epoch, env = env, agent = agent)

      session.lander().filterFinal().logInfo().subscribe()
      session.steps.sample(2 seconds).monitorInfo().subscribe()
      session.run(random)

      logger.info("Session completed.")
    } catch {
      case ex: Throwable =>
        logger.error(ex.getMessage, ex)
        throw ex
    }
  }
}
