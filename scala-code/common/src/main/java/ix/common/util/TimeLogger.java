package ix.common.util;

import org.apache.commons.logging.Log;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.text.DecimalFormat;
import java.util.List;
import java.util.Vector;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 5-apr-2010
 * Time: 17:15:44
 * To change this template use File | Settings | File Templates.
 */
public class TimeLogger {
    public static final int DEFAULT_MAX_N = Integer.MAX_VALUE;
    private int maxN = DEFAULT_MAX_N;
    private Log log;
    private SummaryStatistics stats;
    private String name;
    private final DecimalFormat format = new DecimalFormat("0.000");
    private static boolean GO = true;

    public TimeLogger(Log log, String name) {
        this.log = log;
        this.name = name;
        stats = new SummaryStatistics();
        loggers.add(this);
    }

    public TimeLogger(Log log, String name, int maxN) {
        this.maxN = maxN;
        this.log = log;
        this.name = name;
        stats = new SummaryStatistics();
        loggers.add(this);
    }

    public void addValue(double value) {
        synchronized(stats) {
            stats.addValue(value);
        if (stats.getN()>= maxN)
            reset();
        }
    }

    public void reset() {
        log.info("Time statistics "+name+" (milliseconds): N="+stats.getN()+"; mean="+stats.getMean()+"; std="+stats.getStandardDeviation());
        stats.clear();
    }

    public void reset(int seconds) {
        if (stats.getN() > 0)
            log.info("Time statistics "+name+": SUM="+format.format(stats.getSum())+"("+format.format(stats.getSum()/seconds/10)+"%); N="+format.format(stats.getN())+"; mean="+format.format(stats.getMean())+"; std="+format.format(stats.getStandardDeviation()));
        else
            log.info("Time statistics "+name+": N="+format.format(stats.getN()));
        stats.clear();
    }

    public long start() {
        return System.currentTimeMillis();
    }

    public long pause(long start) {
        return start - System.currentTimeMillis();
    }

    public long restart(long start) {
        return System.currentTimeMillis()-start;
    }

    public void stop(long start) {
        addValue(System.currentTimeMillis()-start);
    }

    private static List<TimeLogger> loggers = new Vector<TimeLogger>();
    public static void init(final int seconds) {
        GO = true;
        new Thread(new Runnable() {
            @Override
            public void run() {
                while(TimeLogger.GO) {
                    try {
                        Thread.currentThread().sleep(1000*seconds);
                    } catch (InterruptedException e) {
                    }
                    for(TimeLogger logger : loggers)
                        logger.reset(seconds);
                }
            }
        }).start();
    }

    public static void stop() {
        GO = false;
    }
}
