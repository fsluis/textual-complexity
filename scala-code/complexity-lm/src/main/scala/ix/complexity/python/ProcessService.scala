package ix.complexity.python

import java.io.IOException

import ix.complexity.core.InstanceService
import ix.complexity.lm.ProcessInstance
import ix.util.services2.CleanableService
import net.liftweb.json.JsonParser.ParseException
import org.apache.commons.configuration.Configuration
import org.apache.commons.logging.{Log, LogFactory}

import scala.util.Try

/**
 * Created by f on 11/08/16.
 */
object ProcessService {
  val log: Log = LogFactory.getLog(classOf[ProcessModel])
}

trait ProcessService extends InstanceService[ProcessModel, ProcessInstance] with CleanableService {

  override def load(name: String, confOpt: Option[Configuration]): ProcessModel = {
    val conf = confOpt.get
    val workDir = conf.getString("work-dir") //, "/local/workspace/com1/altlex2.7"
    val command = conf.getString("command") // , "/local/workspace/com1/altlex2.7/testpipe.sh"
    val waitForInput = conf.getInt("wait-for-input", 1000)
    val initTimeLimit = conf.getInt("time-limit", 60 * 1000) // 1 minute
    val restartEvery = conf.getInt("restart-every", 0)
    val processTimeLimit = conf.getInt("time-out", 10 * 60 * 1000) // 10 minutes
    ProcessModel(workDir, command, waitForInput, initTimeLimit, restartEvery, processTimeLimit)
  }

  override def load(name: String, conf: Option[Configuration], model: ProcessModel): ProcessInstance = {
    val instance = new ProcessInstance(model.workDir, model.command)
    instance.WAIT_FOR_INPUT = model.waitForInput
    instance.INIT_TIME_LIMIT = model.initTimeLimit
    instance.RESTART_EVERY = model.restartEvery
    instance.WAIT_FOR_RESTART = model.processTimeLimit
    instance.init()
    ProcessService.log.info("Done initializing instance, returning: "+instance)
    instance
  }

  override def cleanUp(): Unit = {
    for(instance <- instances())
      instance.cleanUp()
  }

  def analyze(name: String, text: String): Option[String] = {
    val instance = get(name)
    //log.info("Got instance: "+instance)
    val option = try {
      instance.process(text)
    } catch {
      case e: Exception => {
        ProcessService.log.error("Couldn't read input/output for text: "+text, e)
        try { instance.restart() } catch { case e: Exception => ProcessService.log.error("Failed to restart instance after error", e) }
        None }
    } finally {
      //log.info("Returning process instance: "+ instance)
      offer(name, instance)
    }
    option
  }
}

case class ProcessModel(workDir: String, command: String, waitForInput: Int, initTimeLimit: Int, restartEvery: Int, processTimeLimit: Int)
