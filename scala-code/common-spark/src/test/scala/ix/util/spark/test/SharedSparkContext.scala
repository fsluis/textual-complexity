package ix.util.spark.test

import org.scalatest.{Suite, BeforeAndAfterAll}

// C/p'ed from spark test sources (no way to include these via mvn)

/** Shares a local `SparkContext` between all tests in a suite and closes it at the end */
trait SharedSparkContext extends TestSparkContext with BeforeAndAfterAll { self: Suite =>

  override def beforeAll() {
    startSpark()
    super.beforeAll()
  }

  override def afterAll() {
    stopSpark()
    super.afterAll()
  }
}
