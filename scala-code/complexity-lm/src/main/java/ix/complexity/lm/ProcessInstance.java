package ix.complexity.lm;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import scala.Option;
import ix.common.util.TimeLogger;

import java.io.*;
import java.util.Vector;

/**
 * Created by f on 11/08/16.
 */
public class ProcessInstance {
    public Log log = LogFactory.getLog(this.getClass());
    public TimeLogger elog = new TimeLogger(log, "process", 100);

    public Integer WAIT_FOR_INPUT = 100;
    public Integer WAIT_FOR_RESTART = 1000 * 3 * 60; //3 minutes of waiting
    public Integer INIT_TIME_LIMIT = 1000 * 3 * 60; //3 minutes of waiting
    public Integer RESTART_EVERY = 0; //restart every x calls, 0 if none
    public String ENCODING = "UTF-8"; // Command line is usually UTF8
    
    private String command;
    private String homeDir;
    private Process process;
    private boolean initialized = false;
    private BufferedReader input;
    private BufferedWriter output;
    private InputStreamReader reader;
    private ProcessBuilder builder;
    private long counter = 0;

    public ProcessInstance(String homeDir, String command) {
        this.homeDir = homeDir;
        this.command = command;
    }

    private void setProcess(Process process) throws UnsupportedEncodingException {
        this.process = process;
        // From https://stackoverflow.com/questions/8398277/which-encoding-does-process-getinputstream-use
        reader = new InputStreamReader(process.getInputStream(), ENCODING);
        input = new BufferedReader(reader);
        output = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
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
        this.initialized = true;
    }

    public boolean init() throws IOException {
        builder = new ProcessBuilder(command);
        builder.directory(new File(homeDir));
        builder.redirectErrorStream(true);
        System.out.println("Process builder: "+builder.toString());

        System.out.println("Process init: ");
        setProcess(builder.start());
        Thread init = new Thread(new Runnable() {
            public void run() {
                try {
                    String line;
                    while ((line = input.readLine()) != null) {
                        System.out.println("In: "+line);
                        if ("Ready".equals(line)) {
                            setInitialized();
                            System.out.println("Received \"Ready\"");
                            break;
                        }
                    }
                } catch (Exception e) {
                    log.error("Error while waiting for output from process", e);
                }
                System.out.println("Done with initializing process");
            }
        });
        init.start();
        try {
            init.join(INIT_TIME_LIMIT);
        } catch (InterruptedException e) {
            log.error("Error while waiting for output from process", e);
        }
        System.out.println("Done with initializing thread");
        counter=0;
        return isInitialized();
    }

    public boolean restart() throws IOException {
        log.info("Rebooting process after "+counter+" calls");
        try {
            process.destroy();
        } catch (Exception e) {
            log.error("Failed to destroy instance", e);
        }
        return init();
    }

    public Option<String> process(String text) throws IOException, InterruptedException {
        // in case the previous parse didn't go allright, clear the buffers
        while (input.ready())
            input.skip(1);

        if (RESTART_EVERY>0 && counter>=RESTART_EVERY)
            restart();

        long etime;
        String sentence = "";
        etime = elog.start();

        // Write input
        counter +=1;
        writeln(output, "<t>");
        writeln(output, text);
        writeln(output, "</t>");

        // Collect output
        // first, wait for output to appear
        long wait = System.currentTimeMillis();
        while (!input.ready()) {
            if (System.currentTimeMillis() - wait > WAIT_FOR_RESTART) {
                log.error("Rebooting process after timeout");
                //Reboot
                restart();
                wait = System.currentTimeMillis();
            }
            Thread.sleep(WAIT_FOR_INPUT);
        }

        try {
            // then, collect it
            Vector<String> parsed = new Vector<String>();
            while(input.ready()) {
                String line = input.readLine();
                //System.out.println("In: "+line);
                // New text token, check no other lines were received already
                if(line.startsWith("<t>")) {
                    if (parsed.size() != 0)
                        log.warn("New text token received while previous parsed text not returned: " + String.join("\n", parsed));
                    parsed = new Vector<String>();
                } else if(line.endsWith("</t>")) // End of text token, break here
                    break;
                else // Normal line of parsed text
                    parsed.add(line);
            }
            elog.stop(etime);
            return scala.Option.apply(String.join("\n", parsed));
        } catch (Exception e) {
            log.info("Failed to parse output of process: " + e.toString());
        }
        return Option.apply(null);
    }

    private int writeln(BufferedWriter writer, String value) throws IOException {
        //System.out.println("Out: "+value);
        writer.write(value + "\n");
        writer.flush();
        return 1;
    }

    public void cleanUp() {
        try {
            process.destroy();
        } catch (Exception e) {
            log.error("Error while cleaning enju instance");
        }
    }
}
