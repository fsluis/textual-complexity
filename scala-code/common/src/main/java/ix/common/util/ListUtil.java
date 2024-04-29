package ix.common.util;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 13-jan-2010
 * Time: 15:15:50
 * To change this template use File | Settings | File Templates.
 */
public class ListUtil {
    public static List unionHash(List coll1, List coll2) {
        Set union = new HashSet(coll1);
        union.addAll(coll2);
        return new ArrayList(union);
    }

    public static List getUniqueValuesHash(List values) {
        return new Vector(new HashSet(values));
    }

    public static List<? extends Comparable> getUniqueValuesComparable(List<? extends Comparable> values) {
        return new Vector<Comparable>(new TreeSet<Comparable>(values));
    }

    public static Set union(Set<? extends Comparable> coll1, Set<? extends Comparable> coll2) {
        Set union = new TreeSet(coll1);
        union.addAll(coll2);
        return union;
    }

    public static Set<? super Object> intersection(Comparator c, Collection<? super Object> c1, Collection<? super Object> c2) {
        Set<? super Object> intersection = new TreeSet<Object>(c);
        intersection.addAll(c1);
        intersection.retainAll(c2);
        return intersection;
    }

    /**
     * Gives the complement of coll2 to coll1. In other words, the new values in coll2.
     * @param coll1
     * @param coll2
     * @return
     */
    public static Set complement(Set<? extends Comparable> coll1, Set<? extends Comparable> coll2) {
        Set complement = new TreeSet(coll2);
        complement.removeAll(coll1);
        return complement;
    }

    public static List<Integer> indicesOf(List list, Object value) {
        List<Integer> indices = new ArrayList<Integer>();
        for (int i = 0; i < list.size(); i++) {
            if (value.equals(list.get(i)))
                indices.add(i);
        }
        return indices;
    }

    public static double[] listToDoubleArray(List<Double> list) {
        double[] array = new double[list.size()];
        ListIterator<Double> iterator = list.listIterator();
        while (iterator.hasNext())
            array[iterator.nextIndex()] = iterator.next();
        return array;
    }

    public static int[] listToIntegerArray(List<Integer> list) {
        int[] array = new int[list.size()];
        ListIterator<Integer> iterator = list.listIterator();
        while (iterator.hasNext())
            array[iterator.nextIndex()] = iterator.next();
        return array;
    }

    public static double[] integerToDouble(List<Integer> list) {
        double[] array = new double[list.size()];
        ListIterator<Integer> iterator = list.listIterator();
        while (iterator.hasNext())
            array[iterator.nextIndex()] = iterator.next();
        return array;
    }

    public static int sum(int[] array) {
        int sum = 0;
        for (int entry : array)
            sum += entry;
        return sum;
    }

    public static String join(Iterable<?> pColl, String separator) {
        Iterator<?> oIter;
        if (pColl == null || (!(oIter = pColl.iterator()).hasNext()))
            return "";
        StringBuilder oBuilder = new StringBuilder(String.valueOf(oIter.next()));
        while (oIter.hasNext())
            oBuilder.append(separator).append(oIter.next());
        return oBuilder.toString();
    }

    public static List<? super String> fill(List<? super String> list, String s, int n) {
        for (int i=0; i<n; i++)
            list.add(s);
        return list;
    }

    public static boolean contains(Object[] stack, Object needle) {
        for (Object hay : stack)
            if (needle.equals(hay))
                return true;
        return false;
    }

    public static <T> List<T> iteratorToList(Iterator<T> iterator) {
        LinkedList<T> list = new LinkedList<T>();
        while(iterator.hasNext())
            list.add(iterator.next());
        return list;
    }
    
    public static int compare(List<Long> a, List<Long> b) {
        int compare = new Integer(a.size()).compareTo(b.size());
        if(compare!=0) return compare;
        for(int i=0; i<a.size(); i++) {
            compare = a.get(i).compareTo(b.get(i));
            if(compare!=0) return compare;
        }
        return compare;
    }
    
    public static boolean equals(List<Long> a, List<Long> b) {
        return compare(a,b)==0;
    }

    public static double[] doublesToArray(List<Double> list) {
        double[] array = new double[list.size()];
        ListIterator<Double> iterator = list.listIterator();
        while (iterator.hasNext())
            array[iterator.nextIndex()] = iterator.next();
        return array;
    }

    public static int[] intsToArray(List<Integer> list) {
        int[] array = new int[list.size()];
        ListIterator<Integer> iterator = list.listIterator();
        while (iterator.hasNext())
            array[iterator.nextIndex()] = iterator.next();
        return array;
    }
}
