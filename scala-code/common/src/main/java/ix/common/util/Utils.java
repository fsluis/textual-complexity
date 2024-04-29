package ix.common.util;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 1/28/11
 * Time: 10:41 PM
 * To change this template use File | Settings | File Templates.
 */
public class Utils {
    public static String fileExists(String[] files) {
        for(String file : files)
            if (new File(file).exists())
                return file;
        return null;
    }

    public static void writeToFile(String fileName, Serializable object) throws IOException {
        File file = new File(fileName);
         FileOutputStream f = new FileOutputStream(file);
         ObjectOutputStream s = new ObjectOutputStream(f);
         s.writeObject(object);
         s.close();
    }

    public static<S extends Serializable> S readFromFile(File file) throws ClassNotFoundException, IOException {
        FileInputStream f = new FileInputStream(file);
        ObjectInputStream s = new ObjectInputStream(f);
        S object = (S)s.readObject();
        s.close();
        return object;
    }

    public static<S extends Serializable> S readFromFile(String fileName) throws ClassNotFoundException, IOException {
        return readFromFile(new File(fileName));
    }

    public static Path copyFileToTemp(String filename) throws IOException, URISyntaxException {
        File from = new File(filename);
        String tmpDir = System.getProperty("java.io.tmpdir");
        String suffix = "-"+System.currentTimeMillis()+".tmp";
        String tmpFilename = tmpDir+"/"+from.getName()+suffix;
        File tmpFile = new File(tmpFilename);
        //Path tmpPath = Paths.get(new URI( tmpDir+"/"+from.getName()+suffix ));
        //File to = File.createTempFile(from.getName(), tmpDir);
        Path tmpPath = Files.copy(from.toPath(), tmpFile.toPath());
        return tmpPath;
    }

    public static Path writeStringToFile(String filename, String contents) throws IOException {
        return Utils.writeStringToFile(filename, contents, StandardCharsets.UTF_8);
    }

    public static Path writeStringToFile(String filename, String contents, Charset charSet) throws IOException {
        //from https://stackoverflow.com/questions/6879427/scala-write-string-to-file-in-one-statement
        return Files.write(Paths.get(filename), contents.getBytes(charSet));
    }
}
