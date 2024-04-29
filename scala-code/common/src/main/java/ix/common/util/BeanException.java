package ix.common.util;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 12/16/10
 * Time: 1:16 AM
 * To change this template use File | Settings | File Templates.
 */
public class BeanException extends Exception {
    public BeanException() {
        super();    //To change body of overridden methods use File | Settings | File Templates.
    }

    public BeanException(String s) {
        super(s);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public BeanException(String s, Throwable throwable) {
        super(s, throwable);    //To change body of overridden methods use File | Settings | File Templates.
    }

    public BeanException(Throwable throwable) {
        super(throwable);    //To change body of overridden methods use File | Settings | File Templates.
    }
}
