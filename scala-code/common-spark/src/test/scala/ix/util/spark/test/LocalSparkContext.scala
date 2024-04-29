package ix.util.spark.test

import org.scalatest.{Suite, BeforeAndAfterEach}

/** Manages a local `sc` {@link SparkContext} variable, correctly stopping it after each test. */
trait LocalSparkContext extends TestSparkContext with BeforeAndAfterEach { self: Suite =>
  override def beforeEach() {
    startSpark()
    super.beforeEach()
  }

  override def afterEach() {
    stopSpark()
    super.afterEach()
  }
}
