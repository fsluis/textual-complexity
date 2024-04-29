package ix.complexity.features.common

import ix.common.util.MathUtil.log2

case class Entropy(entropy: Double, size: Int) {
  // log2(0) gives -Inf. With an entropy of zero, intuitively the perplexity should also be zero:
  // There's nothing to be perplex about.
  def perplexity = if (entropy > 0) log2(entropy) else 0.0
}
