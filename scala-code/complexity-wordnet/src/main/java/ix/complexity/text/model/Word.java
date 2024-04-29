package ix.complexity.text.model;

//import edu.stanford.nlp.ling.HasWord;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 21-mrt-2010
 * Time: 22:40:25
 * To change this template use File | Settings | File Templates.
 */
public class Word implements IWord { //, HasWord
    String value;

    public Word(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public boolean equals(Object obj) {
        if (obj instanceof Word) {
            Word two = (Word) obj;
            return two.getValue().equalsIgnoreCase(value);
        }
        return super.equals(obj);
    }

    public int compareTo(IWord obj) {
        Word two = (Word) obj;
        return two.getValue().compareToIgnoreCase(value);
    }

    public String toString() {
        return getValue();    //there is almost no (significant) parsing done here.
    }


    public void setValue(String value) {
        this.value = value;
    }

    public Word() {

    }

    //@Override
    public String word() {
        return getValue();
    }

    //@Override
    public void setWord(String s) {
        setValue(s);
    }
}
