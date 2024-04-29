package ix.complexity.stanford;

import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import no.uib.cipr.matrix.LowerSymmBandMatrix;
import no.uib.cipr.matrix.Matrix;

import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * Created by IntelliJ IDEA.
 * User: f
 * Date: 2/18/12
 * Time: 5:14 PM
 * To change this template use File | Settings | File Templates.
 */
public class StanfordCoReferences  {

    public static Matrix corefMatrix(CoreDocument document, int maxK) {
        List<CoreSentence> sentences = document.sentences(); //.get(CoreAnnotations.SentencesAnnotation.class);
        Map<Integer, CorefChain> graph =
                document.corefChains(); //.get(CorefCoreAnnotations.CorefChainAnnotation.class);
        
        //Multiset<Integer> referents = TreeMultiset.create();
        SortedSet<Integer> chainReferents = new TreeSet<Integer>(); //no inner-sentence connections are counted / valued
        Matrix references = new LowerSymmBandMatrix(sentences.size(), maxK);

        //System.out.println("---");
        //System.out.println("coref chains");
        for(CorefChain chain : graph.values()) {
            //System.out.println("\t" + chain);
            chainReferents.clear();
            for(CorefChain.CorefMention mention : chain.getMentionsInTextualOrder()) // Note: replaced .getCorefMentions() (version diff?)
                chainReferents.add(mention.sentNum - 1);   //sentence numbers start at 1, corrected here
            if(chainReferents.size()>1) //only inter-sentence connections are counted
                for (int i : chainReferents)
                    for (int j: chainReferents)
                        if (j<i && j>=(i-maxK)) //{
                            //System.out.println("Setting "+i+","+j);
                            references.add(i, j, 1);
                        //} else
                            //System.out.println("Not setting "+i+","+j);
            //referents.addAll(chainReferents);
            //System.out.println(chainReferents);
        }

        return references;
    }
}
