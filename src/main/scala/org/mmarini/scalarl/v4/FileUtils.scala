package org.mmarini.scalarl.v4

import java.io._

import monix.eval.Task
import monix.reactive.Observable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object FileUtils {

  /**
   * Write data to file
   *
   * @param file   the file
   * @param append true if append write
   * @param f      writer function
   */
  def withFile(file: File, append: Boolean)(f: Writer => Unit) {
    withWriter(new FileWriter(file, append))(f)
  }

  def withWriter(w: Writer)(f: Writer => Unit) {
    try {
      f(w)
    } finally {
      w.close()
    }
  }

  def writeINDArray(matrix: INDArray)(fw: Writer) {
    val Array(n, m) = matrix.shape()
    for {
      i <- 0L until n
    } {
      val record = for {j <- 0L until m} yield matrix.getDouble(i, j).toString
      fw.write(record.mkString(",") + "\n")
    }
  }

  def readINDArray(file: File): Observable[INDArray] =
    readFile(file).map(_.split(",").map(_.toDouble)).foldLeft(Array[Array[Double]]()) {
      case (s, v) => s :+ v
    }.map(Nd4j.create)

  def readFile(file: File): Observable[String] =
    Observable.fromLinesReader(Task.eval(new BufferedReader(new FileReader(file))))
}
