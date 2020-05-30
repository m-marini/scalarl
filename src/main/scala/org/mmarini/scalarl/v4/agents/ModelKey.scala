package org.mmarini.scalarl.v4.agents

case class ModelKey(data: Seq[Int]) {
  lazy val hash = data.hashCode()

  override def hashCode(): Int = hash

  override def equals(obj: Any): Boolean = obj match {
    case x: ModelKey if hash == x.hash => data.equals(x.data)
    case _ => false
  }
}
