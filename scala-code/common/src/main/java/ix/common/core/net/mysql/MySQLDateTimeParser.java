package ix.common.core.net.mysql;

import ix.common.util.TextToolKit;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

/**
 * Created by IntelliJ IDEA on date 28-okt-2005 and time 16:55:03
 *
 * @version 1.0
 */
public class MySQLDateTimeParser {
    public static Calendar parse(String datetime) throws ParseException {
        SimpleDateFormat sdf = new SimpleDateFormat();
        sdf.setCalendar(Calendar.getInstance());
        if (TextToolKit.regexContains(datetime, "\\d\\d\\d\\d-\\d\\d-\\d\\d\\s\\d\\d:\\d\\d:\\d\\d"))
            sdf.applyPattern("yyyy-MM-dd HH:mm:ss");
        else if (TextToolKit.regexContains(datetime, "\\d\\d:\\d\\d:\\d\\d"))
            sdf.applyPattern("HH:mm:ss");
        else if (TextToolKit.regexContains(datetime, "\\d\\d\\d\\d-\\d\\d-\\d\\d"))
            sdf.applyPattern("yyyy-MM-dd");
        else
            throw new ParseException("no valid mysql dateformat: " + datetime, 0);
        sdf.parse(datetime);
        return sdf.getCalendar();
    }
}
