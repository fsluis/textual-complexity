package ix.common.data.wikipedia;

import com.google.common.base.Splitter;
import de.tudarmstadt.ukp.wikipedia.parser.Link;
import de.tudarmstadt.ukp.wikipedia.parser.Paragraph;
import de.tudarmstadt.ukp.wikipedia.parser.Section;
import de.tudarmstadt.ukp.wikipedia.parser.Span;
import ix.common.util.TextToolKit;
import org.apache.commons.lang.StringUtils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Outputs a wikipedia text as paragraphs, text-only.
 */
public class WikipediaParagraphTokenizer {
    private static Pattern TEMPLATE = Pattern.compile("TEMPLATE\\[.*\\]");
    private static Pattern IMAGE = Pattern.compile("\\[\\[Image:.+\\|.+\\|.+\\]\\]");
    private static Pattern FILE = Pattern.compile("\\[\\[File:.+\\|.+\\|.+\\]\\]");
    private static final Splitter SPLITTER = Splitter.on("|");

    public static boolean includeFiles = false;

    public static LinkedHashMap<Long, String> map(Section page) {
        List<Paragraph> paragraphs = page.getParagraphs();
        long n=0;
        LinkedHashMap<Long, String> results = new LinkedHashMap<Long, String>();
        for (Paragraph p : paragraphs) {
            String text = parse(p);
            if(text!=null)
                results.put(n++, text);
        }
        return results;
        //System.out.println("Done parsing page");
    }

    public static String parse(Paragraph p) {
        String text = p.getText();

        List<Span> links = new ArrayList<Span>();
        for (Link link : p.getLinks())
            links.add(link.getPos());
        String linkText = p.getText(links);
        //System.out.println(linkText.length() + "/" + text.length());
        if (linkText.length() >= text.length())
            return null;

        //System.out.println("Paragraph type=" + p.getType() + ", nroftemplates=" + p.getTemplates().size() + ", nroflinks=" + p.getLinks().size());
        text = TEMPLATE.matcher(text).replaceAll("").trim();

        //System.out.println("Before=" + text);
        if (includeFiles) {
            text = replaceImageOrFile(text, IMAGE.matcher(text));
            text = replaceImageOrFile(text, FILE.matcher(text));
        } else {
            text = IMAGE.matcher(text).replaceAll("");
            text = FILE.matcher(text).replaceAll("");
        }

        //System.out.println("After=" + text);
        return text;

        //for(Template t : p.getTemplates())
        //    System.out.println("Template: "+t.getName());

    }

    private static String replaceImageOrFile(String text, Matcher matcher) {
        return TextToolKit.replaceMatches(text, matcher, new TextToolKit.StringReplacer() {
            @Override
            public String replace(String match, int index) {
                Iterator<String> iterator = SPLITTER.split(match).iterator();
                return TextToolKit.replaceTokens(iterator, new TextToolKit.TokenReplacer() {
                    @Override
                    public String replace(String match, int index, boolean last) {
                        if (last)
                            return StringUtils.substring(match, 0, -2);
                        return "";
                    }
                });
            }
        });
    }
}
