package ix.complexity.text.model;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 04-Dec-2010
 * Time: 20:33:42
 * To change this template use File | Settings | File Templates.
 */
public class POSWord extends Word {
    private String tag;

    public POSWord() {
    }

    public POSWord(String value, String tag) {
        super(value);
        this.tag = tag;
    }

    public String getTag() {
        return tag;
    }

    public void setTag(String tag) {
        this.tag = tag;
    }

    public POSWord(String content) {
        super(content);
    }

    public boolean isNoun() {
        return tag.contains("N");
    }

    public boolean isVerb() {
        return tag.contains("V");
    }
}
