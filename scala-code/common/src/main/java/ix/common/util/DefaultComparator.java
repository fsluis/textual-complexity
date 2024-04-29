package ix.common.util;

import java.util.Comparator;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 1/16/11
 * Time: 3:04 PM
 * To change this template use File | Settings | File Templates.
 */
public class DefaultComparator<T> implements Comparator<T> {
    public DefaultComparator() {
    }

    @Override
    public int compare(T x, T y) {
        if (x instanceof Comparable)
            return ((Comparable<T>)x).compareTo(y);
        return new Integer(x.hashCode()).compareTo(y.hashCode());
    }
}

