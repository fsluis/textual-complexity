package ix.util.spark.test

import java.lang.management.ManagementFactory
import ix.util.services2.Service

/**
 * Created with IntelliJ IDEA.
 * Date: 12/1/13
 * Time: 9:40 PM
 * To change this template use File | Settings | File Templates.
 */
object DummyService extends Service {
  val startedByThread = Thread.currentThread().getName + "-id-"+Thread.currentThread().getId+"-hashcode-"+Thread.currentThread().hashCode
  val jvm = ManagementFactory.getRuntimeMXBean.getName
}
