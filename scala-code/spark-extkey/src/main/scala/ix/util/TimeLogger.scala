package ix.util

import scala.Int
import org.apache.commons.logging.Log

class TimeLogger(
    log: Log, name: String, maxN: Int = JTimeLogger.DEFAULT_MAX_N)
  extends JTimeLogger(
    log, name, maxN) {

  def time[That](f: => That): That = {
    val t = start()
    val res = f
    stop(t)
    res
  }
}