package ix.complexity.lm;

import scala.Option;

import java.io.*;
import java.lang.reflect.Field;
import java.util.Iterator;
import java.util.concurrent.*;

/**
 * Created by f on 11/08/16.
 */
public class CommonCrawlPreprocessingInstance {
    private Process process;
    private boolean initialized = false;
    private BufferedReader input;
    private BufferedWriter output;
    private InputStreamReader reader;
    private ProcessBuilder builder;
    private ExecutorService executor = Executors.newSingleThreadExecutor();


    public CommonCrawlPreprocessingInstance() {}

    private void setProcess(Process process) throws UnsupportedEncodingException {
        this.process = process;
        reader = new InputStreamReader(process.getInputStream(), "UTF-8");
        input = new BufferedReader(reader);
        output = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), "UTF-8"));
    }

    public BufferedReader getInput() {
        return input;
    }

    public InputStreamReader getReader() {
        return reader;
    }

    public BufferedWriter getOutput() {
        return output;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public void setInitialized() {
        System.out.println("Done with initializing prepro process");
        this.initialized = true;
    }

    public boolean init() throws IOException {
        builder = new ProcessBuilder("/bin/sh", "-c", "./prepro_post_dedupe_en.sh en | ./prepro_tokenize_en.sh en truecasemodels/truecase-model.en");
        builder.directory( new File(CommonCrawlPreprocessing.preproHome()) );
        builder.redirectErrorStream(true);
        System.out.println("Prepro builder: "+builder.toString());

        System.out.println("Prepro init: ");
        setProcess(builder.start());
        Thread init = new Thread(new Runnable() {
            public void run() {
                try {
                    String line;
                    int lineCount = 0;
                    while (true) {
                        line = getInput().readLine();
                        System.out.println(line);
                        if(++lineCount==6) {
                            // the order of messages seems to fluctuate a bit
                        //if ( line.startsWith("Number of threads:") ) {
                            setInitialized();
                            break;
                        }
                        if (line == null)
                            break;
                        //System.out.print(1);
                    }
                } catch (Exception e) {
                    CommonCrawlPreprocessing.log().error("Error while waiting for output from prepro process", e);
                }
            }
        });
        init.start();
        try {
            init.join(CommonCrawlPreprocessing.initTimeLimit());
        } catch (InterruptedException e) {
            CommonCrawlPreprocessing.log().error("Error while waiting for output from prepro process", e);
        }
        System.out.println("Done with initializing prepro thread");
        return isInitialized();
    }

    public boolean restart() throws IOException {
        try {
            process.destroy();
        } catch (Exception e) {
            CommonCrawlPreprocessing.log().error("Failed to destroy prepro instance", e);
        }
        return init();
    }

    public Option<String> doReduce(String text, int attempt) throws IOException {
        Future<String> f = executor.submit(() -> {
            try {
                return futureReduce(text);
            } catch (IOException e) {
                CommonCrawlPreprocessing.log().info("Failed to parse output of prepro: " + e.toString());
                return null;
            }
        });
        String result = null;
        try {
            result = f.get(30, TimeUnit.SECONDS);
        } catch (ExecutionException | TimeoutException | InterruptedException e) {
            CommonCrawlPreprocessing.log().info("prepro future timeout: " + e.toString());
        }
            if(result==null && ++attempt<3) {
                CommonCrawlPreprocessing.log().error("Rebooting prepro after timeout / failure (attempt: "+attempt+")");
                restart();  // if restart fails, fail whole thing
                doReduce(text, attempt);
            }
        
        return Option.apply(result);
    }


    public String futureReduce(String text) throws IOException, InterruptedException {
        // in case the previous parse didn't go allright, clear the buffers
        while (input.ready())
            input.skip(1);

        long etime;
        etime = CommonCrawlPreprocessing.tlog1().start();
        //write buffer of 10
        write(output, text);

        //read parses
        //long wait = System.currentTimeMillis();
        String response = "";
        String line;
        while ( (line = input.readLine()) != null) {
            response += line + "\n";
            if(line.contains("</P>"))
                break;
        }
            CommonCrawlPreprocessing.tlog1().stop(etime);
            return response;
    }

    private int write(BufferedWriter writer, String value) throws IOException {
        writer.write(value + "\n");
        writer.flush();
        return 1;
    }

    public void cleanUp() {
        try {
            long pid = getPidOfProcess(process);
            Runtime.getRuntime().exec("kill "+pid);
            process.destroyForcibly();
            CommonCrawlPreprocessing.log().info("Destroying instance with pid="+pid+": " + this.toString());
        } catch (Exception e) {
            CommonCrawlPreprocessing.log().error("Error while cleaning prepro instance");
        }
    }

    public long getPid() {
        return getPidOfProcess(process);
    }

    // From https://stackoverflow.com/questions/4750470/how-to-get-pid-of-process-ive-just-started-within-java-program
    public static synchronized long getPidOfProcess(Process p) {
        long pid = -1;

        try {
            if (p.getClass().getName().equals("java.lang.UNIXProcess")) {
                Field f = p.getClass().getDeclaredField("pid");
                f.setAccessible(true);
                pid = f.getLong(p);
                f.setAccessible(false);
            }
        } catch (Exception e) {
            pid = -1;
        }
        return pid;
    }
}
