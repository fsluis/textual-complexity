package ix.complexity.wordnet.model;

import edu.mit.jwi.item.ISynsetID;

import java.util.Comparator;

/**
 * Created by IntelliJ IDEA.
* User: Frans
* Date: 7-apr-2010
* Time: 14:53:24
* To change this template use File | Settings | File Templates.
*/
public class SynsetComparator implements Comparator {
    public int compare(Object o1, Object o2) {
        int diff = 0;
        if ((o1 instanceof ISynsetID) && (o2 instanceof ISynsetID)) {
            ISynsetID sId1 = (ISynsetID) o1;
            ISynsetID sId2 = (ISynsetID) o2;
            int pos1 = sId1.getPOS().getNumber();
            int pos2 = sId2.getPOS().getNumber();
            if (pos1 == pos2)
                diff = sId1.getOffset() - sId2.getOffset();
            else
                diff = pos1 - pos2;
            if (diff > 1) diff = 1;
            if (diff < -1) diff = -1;
        }
        return diff;
    }
}
