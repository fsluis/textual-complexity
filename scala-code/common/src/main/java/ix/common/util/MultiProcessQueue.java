package ix.common.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 1/19/11
 * Time: 3:24 PM
 * To change this template use File | Settings | File Templates.
 */
public class MultiProcessQueue<P> extends ArrayBlockingQueue<P> {
    private static final Log log = LogFactory.getLog(MultiProcessQueue.class);
    private int waitForInstance = 30000;

    public MultiProcessQueue(int capacity) {
        super(capacity);
    }

    public P borrow() {
        P instance = null;
        while (instance == null)
            try {
                instance = poll(waitForInstance, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                log.error("Interrupted while waiting for P instance", e);
            }
        return instance;
    }

    public List<P> drain() {
        List<P> processes = new LinkedList<P>();
        drainTo(processes);
        return processes;
    }

    public void setWaitForInstance(int waitForInstance) {
        this.waitForInstance = waitForInstance;
    }
}
