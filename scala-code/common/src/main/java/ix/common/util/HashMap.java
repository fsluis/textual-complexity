package ix.common.util;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 13-jan-2010
 * Time: 12:30:42
 * To change this template use File | Settings | File Templates.
 */
public class HashMap<K, V> extends LinkedHashMap<K, V> {

    public HashMap(int initialCapacity, float loadFactor) {
        super(initialCapacity, loadFactor);
    }

    public HashMap(int initialCapacity) {
        super(initialCapacity);
    }

    public HashMap() {
    }

    public HashMap(Map<? extends K, ? extends V> m) {
        super(m);
    }
    
    public static String toString(Map map) {
        StringBuilder sb = new StringBuilder();
        for (Object key : map.keySet())
            sb.append(key.toString()).append("=").append(map.get(key).toString()).append(",");
        String result = sb.toString();
        if(result.length()>0)
            return result.substring(0,result.length()-1);
        return "";
    }

    public List<K> findKeys(V value) {
        List<K> keys = new Vector<K>();
        for (Map.Entry<K, V> entry : entrySet()) {
            if (value.equals(entry.getValue()))
                keys.add(entry.getKey());
        }
        return keys;
    }

    /**
     * Returns a new, value-sorted, HashMap. Works only a LinkedHashMap!!!
     * @param comparator
     */
    public void sort(Comparator<Map.Entry<K, V>> comparator) {
        // Get a list of the entries in the map
        List<Map.Entry<K, V>> list = new Vector<Map.Entry<K, V>>(entrySet());

        // Sort the list using an annonymous inner class implementing Comparator for the compare method
        java.util.Collections.sort(list, comparator);

        //Clear
        clear();

        // Copy back the entries now in order
        for (Map.Entry<K, V> entry : list) {
            put(entry.getKey(), entry.getValue());
        }
    }

    public static <K,V extends Comparable<? super V>> LinkedHashMap<K, V> entriesSortedByValues(Map<K, V> map) {
        SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(
            new Comparator<Map.Entry<K,V>>() {
                @Override public int compare(Map.Entry<K,V> e1, Map.Entry<K,V> e2) {
                    return e1.getValue().compareTo(e2.getValue());
                }
            }
        );
        sortedEntries.addAll(map.entrySet());
        LinkedHashMap<K,V> result = new LinkedHashMap();
        Iterator<Map.Entry<K,V>> it = sortedEntries.iterator();
        while(it.hasNext()) {
            Map.Entry<K,V> entry = it.next();
            result.put(entry.getKey(), entry.getValue());
        }

        return result;
    }

    public static <K extends Comparable<? super K>,V extends Comparable<? super V>> LinkedHashMap<K,V> sortHashMapByValues(java.util.HashMap<K,V> passedMap) {
    List<K> mapKeys = new ArrayList(passedMap.keySet());
    List<V> mapValues = new ArrayList(passedMap.values());
    Collections.sort(mapValues);
    Collections.sort(mapKeys);

    LinkedHashMap<K,V> sortedMap =
        new LinkedHashMap<K,V>();

    Iterator<V> valueIt = mapValues.iterator();
    while (valueIt.hasNext()) {
        V val = valueIt.next();
        Iterator<K> keyIt = mapKeys.iterator();

        while (keyIt.hasNext()) {
            K key = keyIt.next();
            V comp1 = passedMap.get(key);
            V comp2 = val;

            if (comp1.equals(comp2)){
                passedMap.remove(key);
                mapKeys.remove(key);
                sortedMap.put(key, val);
                break;
            }

        }

    }
    return sortedMap;
}

}
