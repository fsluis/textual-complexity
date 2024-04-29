package ix.complexity.core

/**
 * Service that holds and serves generic models
 */
trait InstanceService[M,I] extends Instances[M,I] {
  def get(name: String):I = borrow(name)
}
