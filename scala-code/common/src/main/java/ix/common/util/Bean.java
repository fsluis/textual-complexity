package ix.common.util;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 12/15/10
 * Time: 11:52 PM
 * To change this template use File | Settings | File Templates.
 */
public class Bean implements Serializable {
    private Map<Class, List<BeanListener>> listeners = new Hashtable<Class, List<BeanListener>>();

    public Bean() {}

    protected void addEventListener(Class eventClass, BeanListener listenerInstance) {
        if(!listeners.containsKey(eventClass))
            listeners.put(eventClass, new Vector<BeanListener>());
        List<BeanListener> list = listeners.get(eventClass);
        list.add(listenerInstance);
    }

    protected void addEventListener(BeanListener listener) {
        Method[] methods = listener.getClass().getMethods();
        for (Method method : methods)
            if(method.getName().equals("listen")) {
                Class[] parameters = method.getParameterTypes();
                if(parameters.length==1)
                    addEventListener(parameters[0], listener);
            }
    }

    protected void removeEventListener(Class eventClass, BeanListener listenerInstance) {
        if(!listeners.containsKey(eventClass))
            return;
        List<BeanListener> list = listeners.get(eventClass);
        list.remove(listenerInstance);
    }

    protected List<BeanListener> getEventListeners(Class eventClass) {
        if(!listeners.containsKey(eventClass))
            return new Vector<BeanListener>();
        return listeners.get(eventClass);
    }

    protected void propagateEvent(EventObject event) throws BeanException {
        Class eventClass = event.getClass();
        List<BeanListener> listeners = getEventListeners(eventClass);
        if(listeners.size()==0) return;
        Class[] params = {eventClass};
        for(BeanListener listener : listeners)
            try {
                listener.getClass().getMethod("listen", params).invoke(listener, event);
            } catch (Exception e) {
                throw new BeanException("Couldn't invoke listener "+listener.toString()+" for event type "+eventClass, e);
            }
    }


}
