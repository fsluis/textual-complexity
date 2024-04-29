package ix.data.structure

case class Text(plainText: String, charIndex: Long = 0) extends Content {
  override def compare(that: Content): Int = super.compareTo(that)

  override def toString: String = s"Text(index:$charIndex, length:${plainText.length}, content:$plainText)"
}
