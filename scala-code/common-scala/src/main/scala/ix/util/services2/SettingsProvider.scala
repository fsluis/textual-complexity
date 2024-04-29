package ix.util.services2

import org.apache.commons.configuration.HierarchicalConfiguration

/**
 * Created with IntelliJ IDEA.
 * User: f
 * Date: 12/1/13
 * Time: 4:03 PM
 * To change this template use File | Settings | File Templates.
 */
trait SettingsProvider {
  // Note: the returned configuration object is mut(at)ed
  def getConfiguration: HierarchicalConfiguration

  // See http://www.ibm.com/developerworks/library/j-5things12/
  // and https://github.com/francisdb/serviceloader-maven-plugin
}
