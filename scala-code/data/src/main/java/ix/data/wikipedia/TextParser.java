package ix.data.wikipedia;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import de.tudarmstadt.ukp.wikipedia.parser.Link;
import de.tudarmstadt.ukp.wikipedia.parser.Paragraph;
import de.tudarmstadt.ukp.wikipedia.parser.Span;
import ix.common.util.TextToolKit;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.StringEscapeUtils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Outputs a wikipedia text as paragraphs, text-only.
 * Todo: merge with WikipediaParser service
 */
public class TextParser {
    private static Pattern TEMPLATE = Pattern.compile("TEMPLATE\\[.*\\]");
    private static Pattern IMAGE = Pattern.compile("\\[\\[Image:.+\\|.+\\|.+\\]\\]");
    private static Pattern FILE = Pattern.compile("\\[\\[File:.+\\|.+\\|.+\\]\\]");
    private static final Splitter SPLITTER = Splitter.on("|");

    // Added 16-10-2019
    private static Pattern CATEGORY = Pattern.compile("\\[\\[Category:.+\\]\\]");
    private static Pattern LANG = Pattern.compile("\\[\\[[a-z]{2}:.+\\]\\]");
    private static Pattern GOT = Pattern.compile("\\[\\[got:$");

    public static boolean includeFiles = false;

    /**
     * Can return null! Parses paragraph to (clean) string
     * @param p
     * @return Null or clean text
     */
    public static String parse(Paragraph p) {
        String text = p.getText();
        text = StringEscapeUtils.unescapeHtml4(text);

        List<Span> links = new ArrayList<Span>();
        for (Link link : p.getLinks())
            links.add(link.getPos());
        String linkText = p.getText(links);
        //System.out.println(linkText.length() + "/" + text.length());
        if (linkText.length() >= text.length())
            return null;

        // added 19-10-2019
        for(Link link: p.getLinks()) {
            String link_text = link.getText();
            String anchor_text = Iterables.getLast(SPLITTER.split(link_text));
            text = text.replace(link_text, anchor_text);
        }

        //System.out.println("Paragraph type=" + p.getType() + ", nroftemplates=" + p.getTemplates().size() + ", nroflinks=" + p.getLinks().size());
        text = TEMPLATE.matcher(text).replaceAll("").trim();

        //System.out.println("Before=" + text);
        if (includeFiles) {
            text = replaceImageOrFile(text, IMAGE.matcher(text));
            text = replaceImageOrFile(text, FILE.matcher(text));
        } else {
            text = IMAGE.matcher(text).replaceAll("");
            text = FILE.matcher(text).replaceAll("");
            text = CATEGORY.matcher(text).replaceAll("");
            text = LANG.matcher(text).replaceAll("");
            text = GOT.matcher(text).replaceAll("");
        }

        //System.out.println("After=" + text);
        return text;

        //for(Template t : p.getTemplates())
        //    System.out.println("Template: "+t.getName());

    }

    private static String replaceImageOrFile(String text, Matcher matcher) {
        return TextToolKit.replaceMatches(text, matcher, new TextToolKit.StringReplacer() {
            public String replace(String match, int index) {
                Iterator<String> iterator = SPLITTER.split(match).iterator();
                return TextToolKit.replaceTokens(iterator, new TextToolKit.TokenReplacer() {
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
