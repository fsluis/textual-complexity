package ix.util.services2

/**
 * Special-purpose Service that holds and serves generic data models. The served data models need to be reusable
 * and multi-threaded. If not, see {@link ix.complexity.InstanceService InstanceService}.
 *
 * @tparam M Data model type
 */
trait ModelService[M] extends Models[M] {
  def get(name: String = Models.DEFAULT):M = getModel(name)
  //def apply(model: String) = get(model)
    //Models.DEFAULT < this gave a compiler error when overloeding and setting the default there
  // and in general this method created confusion during compiling down the road
}
