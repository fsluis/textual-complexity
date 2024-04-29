package ix.common.util;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 1/14/11
 * Time: 3:26 PM
 * To change this template use File | Settings | File Templates.
 */
public class MathUtil {

    public static double log2 = Math.log(2);

    public static double log2(double number) {
        return Math.log(number) / log2;
    }
}
