package ix.complexity.core

//import scala.actors.threadpool.{TimeUnit, LinkedBlockingQueue, BlockingQueue}

import java.lang.management.ManagementFactory
import java.util.concurrent.{BlockingQueue, ConcurrentHashMap, LinkedBlockingQueue, TimeUnit}

import com.sun.management.OperatingSystemMXBean
import ix.complexity.core.Instances.log
import ix.util.services2.{Models, ServiceException}
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}

import scala.collection.JavaConversions._
import scala.compat.java8.FunctionConverters._

trait Instances[M, I] extends Models[M] {
  val queues = new ConcurrentHashMap[String, BlockingQueue[InstanceWrapper[I]]]
  /*val queues = new mutable.HashMap[String, BlockingQueue[I]] with mutable.SynchronizedMap[String, BlockingQueue[I]] {
    def mkGet(key: String): BlockingQueue[I] = {
      if(!contains(key)) put( key, mkQueue(key) )
      get(key).get
    }
  } */
  // from https://stackoverflow.com/questions/25552/get-os-level-system-information
  val os: com.sun.management.OperatingSystemMXBean = ManagementFactory.getOperatingSystemMXBean.asInstanceOf[OperatingSystemMXBean]
  val maxPerMB = math.round(os.getTotalPhysicalMemorySize / 1024 / 1024 / config.getDouble("min-mem-mb-per-instance", 1d)).toInt
  val maxPerCore = math.round(Runtime.getRuntime.availableProcessors * config.getDouble("max-instances-per-core", 1d)).toInt
  val maxInstances = List(1, List(config.getInt("max-instances", 100),maxPerCore,maxPerMB).min).max
  val waitForInstance = config.getInt("wait-for-instance", 30000)

  def instances(): Iterable[I] =
    queues.flatMap(_._2.flatMap(_.instance))

  // this is needed for computeIfAbsent on the concurrenthashmap implementation of java. It calls the function when
  // a key is not available in the map, in order to create the value for that key.
  val lambda: String => BlockingQueue[InstanceWrapper[I]] = name => {
    val queue = new LinkedBlockingQueue[InstanceWrapper[I]](maxInstances)
    log.info("Creating instance queue for " + name + ", maxInstances: " + maxInstances)
    for (i <- 0 until maxInstances) {
      /*log.info("Loading " + getClass.getSimpleName + " instance " + name + " nr " + i)
      val conf = getConfig(name)
      val model = getModel(name)
      // .getOrElse(throw new ServiceException("Unknown model " + name+" for service "+getClass.getSimpleName+". Available models: "+configurations.keySet.mkString(", ")+"."))
      if(conf.isEmpty) log.warn("Unknown model " + name+" requested for service "+getClass.getSimpleName+". Available models: "+configurations.keySet.mkString(", ")+".")
      val instance = load(name, conf, model)
      queue.add(instance)*/
      queue.add(new InstanceWrapper[I](name, lambdaStartInstance))
    }
    //log.info("Added "+queue.size()+" instances for "+name+" to queue@"+ Integer.toHexString(queue.hashCode) +" "+queue+" for this: "+this.toString)
    queue
  }

  // This was needed to have a wrapper around an instance, for delayed starting of the instance (on borrow)
  val lambdaStartInstance: String => I = name => {
    log.info("Loading " + getClass.getSimpleName + " instance " + name)
    val conf = getConfig(name)
    val model = getModel(name)
    // .getOrElse(throw new ServiceException("Unknown model " + name+" for service "+getClass.getSimpleName+". Available models: "+configurations.keySet.mkString(", ")+"."))
    if(conf.isEmpty) log.warn("Unknown model " + name+" requested for service "+getClass.getSimpleName+". Available models: "+configurations.keySet.mkString(", ")+".")
    load(name, conf, model)
  }

  // Todo: change contract of load to Option[Configuration] and Option[I]
  // So some implementations can still return an Instance even if there's no Conf given
  def load(name: String, conf: Option[Configuration], model: M): I

  def borrow(name: String): I = {
    //val test = lambda.asJava.apply(name)
    //log.info("Test lambda output: "+test)
    val queue: BlockingQueue[InstanceWrapper[I]] = queues.computeIfAbsent(name, lambda.asJava)
    var wrapper: Option[InstanceWrapper[I]] = None
    while (wrapper.isEmpty) try {
      wrapper = Option(queue.poll(waitForInstance, TimeUnit.MILLISECONDS))
        if (wrapper.isEmpty) {
          log.warn("Unsuccessfully waited "+waitForInstance+"ms for an instance of "+name+". Will continue waiting.")
          //log.warn("Using queue for "+name+", size "+queue.size+", queue@"+ Integer.toHexString(queue.hashCode)+": "+queue+", this: "+this.toString)
        }
      } catch {
        case e: InterruptedException => log.error("Interrupted while waiting for instance", e)
      }
    wrapper.get.get()
  }

  def offer(name: String, instance: I) =
    queues.get(name).add( InstanceWrapper(name, lambdaStartInstance, Some(instance) ))

  def cleanUp(name: String, callback: I => Unit) = {
    val queue = queues.remove(name)
    if(queue!=null) {
      log.info("Cleaning up queue for " + name + ", maxInstances: " + maxInstances)
      queue.foreach( _.instance.map( callback(_) ) )
      log.info("Done with stopping instances for " + name)
    } //else
      //log.warn("Tried to stop instances but no queue found for "+name)
  }
}

object Instances {
  private final val log: Log = LogFactory.getLog(this.getClass)
}

case class InstanceWrapper[I](name: String, start: String => I, instance: Option[I] = None) {
  def get(): I = instance.getOrElse(start(name))
}