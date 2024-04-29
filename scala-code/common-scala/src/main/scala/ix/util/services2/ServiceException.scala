package ix.util.services2

class ServiceException(str: String, thr: Throwable) extends Exception(str, thr) {
  def this(str: String) = this(str, null)
}