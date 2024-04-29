package ix.common.util;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 4/8/11
 * Time: 4:52 PM
 * To change this template use File | Settings | File Templates.
 */
public class LimitedCapacityHashMap<K, V> extends LinkedHashMap<K, V> {
    public static final int DEFAULT_CAPACITY = 100;
    private int capacity;

    public LimitedCapacityHashMap(int capacity) {
        //capacity + 1, just to be sure no rehashing operations are to occur
        //access-order lookup: cache based on recency of lookups (and writes)
        super(capacity+1, 1, true);
        this.capacity = capacity;
    }

    public LimitedCapacityHashMap() {
        super(DEFAULT_CAPACITY);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry eldest) {
        return size() > capacity;
    }

    public int getCapacity() {
        return capacity;
    }

    //public void setCapacity(int capacity) {
    //    this.capacity = capacity;
    //}
    
    public boolean isFull() {
        return size() >= capacity;
    }

    public Map.Entry<K, V> eldest() {
        return entrySet().iterator().next();
    }

    public Map.Entry<K, V> removeEldest() {
        Map.Entry<K, V> entry = eldest();
        remove(entry.getKey());
        return entry;
    }
    
    public Map.Entry<K, V> reduceCapacity() {
        Map.Entry<K, V> entry = null;
        if(isFull())
            entry = removeEldest();
        capacity--;
        return entry;
    }
    
    public int increaseCapacity() {
        return ++capacity;
    }

    public void clear(int capacity) {
        super.clear();
        this.capacity = capacity;
    }
}
