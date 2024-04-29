package ix.util.spark

import ix.util.services2.SettingsProvider
import org.apache.commons.configuration.HierarchicalConfiguration


/**
 * Created with IntelliJ IDEA.
 * Date: 12/1/13
 * Time: 5:48 PM
 * To change this template use File | Settings | File Templates.
 */
class SparkSettingsProvider extends SettingsProvider {
  def getConfiguration: HierarchicalConfiguration = {
    new HierarchicalConfiguration()
  }
}
