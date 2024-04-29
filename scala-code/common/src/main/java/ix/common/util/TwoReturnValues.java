package ix.common.util;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 5-apr-2010
 * Time: 2:12:36
 * To change this template use File | Settings | File Templates.
 */
public final class TwoReturnValues<R,S> {
	public final R first;
	public final S second;
	
	public TwoReturnValues(R first, S second) {
		this.first = first;
		this.second = second;
	}

	public R getFirst() {
		return first;
	}

	public S getSecond() {
		return second;
	}
}
