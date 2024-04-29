package ix.common.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.File;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 1/3/12
 * Time: 2:00 PM
 * To change this template use File | Settings | File Templates.
 */
public class FileUtils extends Utils {
    private static final Log log = LogFactory.getLog(FileUtils.class);
    
    public static List<File> getFilesRecursive(String dirName, Pattern pattern) {
        List<File> files = new LinkedList<File>();
        File dir = new File(dirName);
        if (dir.isDirectory()) {
            traverseDirectory(files, dir, pattern);
            log.info("Found "+files.size()+" files");
        } else
            log.error("DirName " + dirName + " should point to a directory!");
        return files;
    }

    private static void traverseDirectory(List<File> files, File dir, Pattern pattern) {
        for(File file : dir.listFiles())
            if(file.isDirectory())
                traverseDirectory(files, file, pattern);
            else if(pattern == null || pattern.matcher(file.getName()).matches())
                files.add(file);
    }

}
