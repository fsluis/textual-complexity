package ix.data.structure

import ix.data.com.datasets.EnSiWikiDataSet
import ix.util.services2.ModelService
import org.apache.commons.configuration.Configuration

object DataSetService extends ModelService[DataSet] {
  override def load(name: String, confOpt: Option[Configuration]): DataSet = {
    val conf = confOpt.get
    val classname = conf.getString("classname", classOf[EnSiWikiDataSet].getCanonicalName )
    val dataset: DataSet = Class.forName(classname).newInstance.asInstanceOf[DataSet]
    dataset.init(name,conf)
    dataset
  }

  def apply(name: String): DataSet = get(name)
}
