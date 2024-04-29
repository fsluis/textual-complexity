package ix.data.structure

trait Content extends Serializable with Ordered[Content] {
  def plainText: String

  /**
   * Character-based index for this content.
   * Note that this index does not necessarily align with the plainText lengths, as the latter adds seperators (eg \n\n)
   * between content items.
   * @return character index
   */
  def charIndex: Long

  override def compareTo(that: Content): Int = (this.charIndex - that.charIndex).toInt
}
