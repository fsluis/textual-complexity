package ix.common.util;

import java.util.LinkedList;

/**
 * FIFO implementation with maximum size
 */
public class LimitedCapacityQueue<E> extends LinkedList<E> {
        private int capacity = 100;

        public LimitedCapacityQueue() {
        }

        public LimitedCapacityQueue(int capacity) {
            this.capacity = capacity;
        }

    /**
     * Push item onto the end of the queue, removing the first item of the queue if over-capacity.
      * @param item
     * @return if over-capacity, the item removed is returned
     */
    public E append(E item) {
            E removed = null;
            super.add(item);    //To change body of overridden methods use File | Settings | File Templates.
            if (this.size()>capacity)
                removed = this.remove();
            return removed;
        }

    public E find(Object item) {
        int index = this.lastIndexOf(item);
        if(index==-1)
            return null;
        else
            return this.get(index);
    }
}
