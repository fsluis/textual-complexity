package ix.complexity.wordnet.model;

import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.*;
import edu.mit.jwi.morph.IStemmer;
import edu.mit.jwi.morph.WordnetStemmer;
import ix.complexity.text.model.IWord;
import ix.complexity.text.model.POSWord;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.*;

import static edu.mit.jwi.data.ILoadPolicy.IMMEDIATE_LOAD;
import static ix.common.util.Utils.fileExists;

/**
 * Created by IntelliJ IDEA.
 * User: Frans
 * Date: 7-apr-2010
 * Time: 16:18:49
 * To change this template use File | Settings | File Templates.
 */
public class WordNetUtil2 {
    private static String[] model = {"resources/WNdb-3.0", "WNdb-3.0"};
    private static IDictionary dict;
    private static IStemmer stemmer;

    public WordNetUtil2() throws IOException {
        init();
    }

    public WordNetUtil2(String model) throws IOException {
        setModel(model);
        init();
    }

    public static String getModel() {
        return model[0];
    }

    public static void setModel(String model) {
        WordNetUtil2.model = new String[]{model};
    }

    public void init() throws IOException {
        if (dict == null) {
            String path = fileExists(model) + File.separator + "dict";
            URL url = new URL("file", null, path);
            //dict = new edu.mit.jwi.Dictionary(url);
            dict = new edu.mit.jwi.RAMDictionary(url, IMMEDIATE_LOAD);
            dict.open();
            stemmer = new WordnetStemmer(dict);
        }
    }

    public static IDictionary getDict() {
        return dict;
    }

    public static IStemmer getStemmer() {
        return stemmer;
    }

    public SortedSet<ISynsetID> getRelatedSynsets(Collection<ISynsetID> synsets, IPointer pointer) {
        SortedSet<ISynsetID> related = new TreeSet<ISynsetID>(new SynsetComparator());
        for (ISynsetID synset : synsets) {
            related.addAll(getRelatedSynsets(synset, pointer));
        }
        return related;
    }

    public SortedSet<ISynsetID> getRelatedSynsetsWithoutHyponymy(ISynsetID synset) {
        SortedSet<ISynsetID> related = new TreeSet<ISynsetID>(new SynsetComparator());
        related.addAll(getRelatedSynsets(synset));
        related.removeAll(getRelatedSynsets(synset, Pointer.HYPONYM)); // Cf. Gervasi2003: no hyponymy links
        return related;
    }

    public SortedSet<ISynsetID> getRelatedSynsets(ISynsetID synsetId) {
        return getRelatedSynsets(synsetId, null);
    }

    public SortedSet<ISynsetID> getRelatedSynsets(ISynsetID synsetId, IPointer pointer) {
        SortedSet<ISynsetID> related = new TreeSet<ISynsetID>(new SynsetComparator());
        ISynset synset = dict.getSynset(synsetId);
        if (pointer==null)
            related.addAll(synset.getRelatedSynsets());
        else
            related.addAll(synset.getRelatedSynsets(pointer));
        return related;
    }

    public SortedSet<ISynsetID> findAllSynsets(ix.complexity.text.model.IWord word) {
        SortedSet<ISynsetID> synsets = new TreeSet<ISynsetID>(new SynsetComparator()); 
        for ( POS pos : POS.values())
            synsets.addAll(findSynsets(word, pos));
        return synsets;
    }

    public SortedSet<ISynsetID> findSynsets(String stem, List<POS> poses) {
        SortedSet<ISynsetID> synsets = new TreeSet<ISynsetID>(new SynsetComparator());
        for ( POS pos : poses)
            synsets.addAll(findSynsets(stem, pos));
        return synsets;
    }

    public SortedSet<ISynsetID> findSynsets(POSWord word) {
        if(word.isNoun())
            return findSynsets(word, POS.NOUN);
        else if (word.isVerb())
            return findSynsets(word, POS.VERB);
        return findAllSynsets(word);
    }

    public SortedSet<ISynsetID> findSynsets(ix.complexity.text.model.IWord word, POS pos) {
        return findSynsets(getStems(word, pos), pos);
    }

    public SortedSet<ISynsetID> findSynsets(Collection<String> stems, POS pos) {
        SortedSet<ISynsetID> synsets = new TreeSet<ISynsetID>(new SynsetComparator());
        for (String stem : stems)
            synsets.addAll(findSynsets(stem, pos));
        return synsets;
    }

    private SortedSet<ISynsetID> findSynsets(String stem, POS pos) {
        SortedSet<ISynsetID> synsets = new TreeSet<ISynsetID>(new SynsetComparator());
            IIndexWord indexWord = dict.getIndexWord(stem, pos);
            if (indexWord != null) {
                List<IWordID> words = indexWord.getWordIDs();
                for (IWordID wordID : words) {
                    synsets.add(dict.getWord(wordID).getSynset().getID());
                }
            }
        return synsets;
    }

    /*public SortedSet<ISynsetID> findAllSynsets(ISentence sentence) {
        return findAllSynsets(sentence.getWords());
    }*/

    /*public SortedSet<ISynsetID> findAllSynsets(ISentence sentence, POS pos) {
        return findAllSynsets(sentence.getWords(), pos);
    }*/

    public SortedSet<ISynsetID> findSynsets(List<ix.complexity.text.model.IWord> soup) {
        SortedSet<ISynsetID> synsets = new TreeSet<ISynsetID>(new SynsetComparator());
        SortedSet<ix.complexity.text.model.IWord> words = new TreeSet<ix.complexity.text.model.IWord>();
        words.addAll(soup);
        for (ix.complexity.text.model.IWord word : words)
            synsets.addAll(findSynsets(word, POS.NOUN));
        return synsets;
    }

    public List<String> getStems(ix.complexity.text.model.IWord word, POS pos) {
        return getStems(word.toString(), pos);
    }

    public Set<String> getStems(List<ix.complexity.text.model.IWord> words, POS pos) {
        Set<String> stems = new HashSet<String>();
        for (IWord word : words)
            stems.addAll(getStems(word.toString(), pos));
        return stems;
    }

    public List<String> getStems(String word, POS pos) {
        return stemmer.findStems(word, pos);
    }

    public static SortedSet<ISynsetID> intersection(SortedSet<ISynsetID> sy1, Collection<ISynsetID> sy2) {
        SortedSet<ISynsetID> intersection = new TreeSet(sy1);
        //System.out.println("comparator: "+intersection.comparator());
        intersection.retainAll(sy2);
        return intersection;
    }
}
