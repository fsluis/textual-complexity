package ix.data.structure

/**
 * Content container (for eg article, sections, subsections, etc)
 * @param sections Note that this list should not be empty! Except for the top level. Otherwise the charIndex gets messed up.
 */
class ContentList(sections: Seq[Content]) extends Content {
  val sorted: Seq[Content] = sections.sorted

  override def plainText: String = sorted.map(_.plainText).mkString("\n\n")

  override def charIndex: Long = sorted.headOption.map(_.charIndex).getOrElse(0)

  /**
   * Last character-based index of this content list.
   * @return Exclusive (eg char-string "the" gives lastIndex of 2)
   */
  def lastIndex: Long = sorted.lastOption.map( c => c.charIndex+c.plainText.length-1 ).getOrElse(0)

  override def compare(that: Content): Int = super.compareTo(that)

  override def toString: String = s"ContentList(index:$charIndex-$lastIndex,size:${sorted.size})"
}
