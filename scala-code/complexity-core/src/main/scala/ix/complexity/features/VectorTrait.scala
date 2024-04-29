package ix.complexity.features

trait VectorTrait {
  def cosine(other: VectorTrait): Double
  def values: Array[Double]
  def entropy: Double
}
