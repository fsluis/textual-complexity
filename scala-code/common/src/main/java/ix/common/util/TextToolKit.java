package ix.common.util;

import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * this class contains all sorts of handy functions if you work a lot with Strings.
 *@author f
 *@author m
 *@version 1.1
 */

public class TextToolKit {
	public static String randomString(int length) {
		StringBuffer buf = new StringBuffer(length);
		Random rnd = new Random();
		int nextChar;
		int range = 'z' - 'a' + 1;
		for (int i = 0; i < length; i++) {
			nextChar = 'a' + rnd.nextInt(range);
			buf.append((char) nextChar);
		}
		return buf.toString();
	}

	/**
	 * replaces a string in a text.
	 *@param haystack the text to search through
	 *@param needle the text to find (replace)
	 *@param replacement the text to put on the place of the needle
	 */
	public static String stringReplace(String haystack, String needle, String replacement) {
		if (needle.length() == 0) return haystack;
		String newString = new String(haystack);

		int offSet = newString.indexOf(needle);
		while (offSet != -1) {
			StringBuffer helper = new StringBuffer(newString);
			helper.replace(offSet, offSet + needle.length(), replacement);
			newString = helper.toString();
			offSet = offSet + replacement.length();
			offSet = newString.indexOf(needle, offSet);
		}
		return newString; // nothing to replace
	}

	/**
	 * does a string replace based on a regular expression
	 *@param pattern the pattern to match
	 *@param replacement the replacement of the pattern
	 *@param text the text to execute the replacement on
	 *@since 1.1
	 */
	public static String regularReplace(String pattern, String replacement, String text) {
		Pattern regularPattern = Pattern.compile(pattern);
		Matcher matcher = regularPattern.matcher(text);
		return matcher.replaceAll(replacement);
	}

	public static int stringCount(String haystack, String needle) {
		int count = -1, last = -1;
		do {
			count++;
			last = haystack.indexOf(needle, last+1);
		} while (last != -1);
		return count;
	}

	public static String[] regexFind(String haystack, String regex) {
		Matcher matcher = Pattern.compile(regex).matcher(haystack);
		Vector results = new Vector();
		while (matcher.find())
			results.add(matcher.group());
		return (String[]) results.toArray(new String[0]);
	}

	public static String regexFindFirst(String haystack, String regex) {
		Matcher matcher = Pattern.compile(regex).matcher(haystack);
		if (matcher.find())
			return matcher.group();
		return null;
	}

	public static String regexFindLast(String haystack, String regex) {
		Matcher matcher = Pattern.compile(regex).matcher(haystack);
		String result = new String();
		while (matcher.find())
			result = matcher.group();
		return result;
	}

	public static boolean regexContains(String haystack, String regex) {
		Matcher matcher = Pattern.compile(regex).matcher(haystack);
		return matcher.lookingAt();
	}

	public static String join(String[] args) {
		return join(args, " ");
	}

    public static String join(String[] args, String glue) {
        String joined = "";
        for (int i = 0; i < args.length; i++) {
            if (i != 0)
                joined = joined + glue;
            joined = joined + args[i];
        }
        return joined;
    }

	public static String textToHtml(String text) {
		text = stringReplace(text, "&", "&amp;");
		text = stringReplace(text, ">", "&gt;");
		text = stringReplace(text, "<", "&lt;");
		text = stringReplace(text, "\"", "&quot;");
		text = stringReplace(text, "\n", "<br />");
		return text;
	}

	public static String htmlToText(String html) {
		html = stringReplace(html, "<br />", "\n");
		html = stringReplace(html, "&quot;", "\"");
		html = stringReplace(html, "&lt;", "<");
		html = stringReplace(html, "&gt;", ">");
		html = stringReplace(html, "&amp;", "&");
		return html;
	}

	public static String getHtmlChar(String html, int index) {
		String text = htmlToText(html);
		text = text.substring(index, index + 1);
		return textToHtml(text);
	}

	/**
	 * Chops this String to the given needle, searching for the needle
	 * from the given position in the given String.
	 * @param in the String to chop
	 * @param from the position from where to search for the needle
	 * @param needle the needle to which will be chopped
	 */
	public static String chopTo(String in, int from, String needle) {
		int index = in.indexOf(needle, from);
		if (index == -1) return in;
		return in.substring(index);
	}

	/**
	 * Chops this String to the given needle.
	 * @param in the String to chop
	 * @param needle the needle to which will be chopped
	 */
	public static String chopTo(String in, String needle) {
		return chopTo(in, 0, needle);
	}

	/**
	 * Chops this String to and including the given needle, searching for the needle from
	 * the given position in the given String.
	 * @param in the String to chop
	 * @param from the position from where to search for the needle
	 * @param needle the needle to and including which will be chopped
	 */
	public static String chopToInclusive(String in, int from, String needle) {
		int index = in.indexOf(needle, from);
		if (index == -1) return in;
		return in.substring(index + needle.length());
	}

	/**
	 * Chops this String to and including the given needle.
	 * @param in the String to chop
	 * @param needle the needle to and including which will be chopped
	 */
	public static String chopToInclusive(String in, String needle) {
		return chopToInclusive(in, 0, needle);
	}

	/**
	 * Chops this String from after the given needle, searching for the needle
	 * from the given position in the given String.
	 * @param in the String to chop
	 * @param from the position from where to search for the needle
	 * @param needle the needle frp, which will be chopped
	 */
	public static String chopFrom(String in, int from, String needle) {
		int index = in.indexOf(needle, from);
		if (index == -1) return in;
		return in.substring(0, index + needle.length());
	}

	/**
	 * Chops this String from after the given needle.
	 * @param in the String to chop
	 * @param needle the needle frp, which will be chopped
	 */
	public static String chopFrom(String in, String needle) {
		return chopFrom(in, 0, needle);
	}

	/**
	 * Chops this String from and including the given needle, searching for the needle from
	 * the given position in the given String.
	 * @param in the String to chop
	 * @param from the position from where to search for the needle
	 * @param needle the needle to and including which will be chopped
	 */
	public static String chopFromInclusive(String in, int from, String needle) {
		int index = in.indexOf(needle, from);
		if (index == -1) return in;
		return in.substring(0, index);
	}

	/**
	 * Chops this String from and including the given needle.
	 * @param in the String to chop
	 * @param needle the needle to and including which will be chopped
	 */
	public static String chopFromInclusive(String in, String needle) {
		return chopFromInclusive(in, 0, needle);
	}

	/**
	 * Looks if one of the Strings in haystack equals needle, if so
	 * returns true, otherwise returns no. Compares using the String it's
	 * equal method.
	 * @param needle the needle to compare with.
	 * @param haystack the Strings to look in for the needle.
	 */
	public static boolean equals(String needle, String[] haystack) {
		for (int i = 0; i < haystack.length; i++)
			if (haystack[i].equals(needle))
				return true;
		return false;
	}

	/**
	 * Returns a string with backslashes before characters that need
	 * to be quoted in database queries etc. These characters are single
	 * quote ('), double quote (") and backslash (\).
	 * @return the backslashed string
	 */
	public static String addSlashes(String in) {
		in = stringReplace(in, "\\", "\\\\");
		in = stringReplace(in, "'", "\\'");
		in = stringReplace(in, "\"", "\\\"");
		return in;
	}

	public static String stripSlashes(String in) {
		in = stringReplace(in, "\\\"", "\"");
		in = stringReplace(in, "\\'", "'");
		in = stringReplace(in, "\\\\", "\\");
		return in;
	}

    // Also look at ListUtil.join()
    public static String implode(Object[] ary, Object delim) {
        return implode(Arrays.asList(ary).iterator(), delim);
    }

    public static String implode(Iterator values, Object delim) {
        StringBuilder out = new StringBuilder();
        while(values.hasNext()) {
            out.append(values.next());
            if(values.hasNext())
                out.append(delim);
        }
        return out.toString();
    }

    public static boolean equalsIgnoreCase(Collection<String> c1, Collection<String> c2) {
        for (String s1 : c1) {
            for (String s2 : c2) {
                if (s1.equalsIgnoreCase(s2)) return true;
            }
        }
        return false;
    }


    public static String replaceMatches(String original, Matcher matcher, StringReplacer replacer) {
        if (!matcher.find())
            return original;
        StringBuffer sb = new StringBuffer();
        int i=0;
        do {
            String match = matcher.group();
            String replacement = replacer.replace(match, i++);
            matcher.appendReplacement(sb, replacement);
        } while (matcher.find());
        matcher.appendTail(sb);
        return sb.toString();
    }

    public static String replaceTokens(Iterator<? extends String> tokens, TokenReplacer replacer) {
        StringBuilder sb = new StringBuilder();
        int i=0;
        while (tokens.hasNext())
            sb.append(replacer.replace(tokens.next(), i++, !tokens.hasNext()));
        return sb.toString();
    }

    public static interface StringReplacer {
        public String replace(String match, int index);
    }

    public static interface TokenReplacer {
        public String replace(String match, int index, boolean last);
    }

    public static String join(String token, int times) {
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<times; i++)
            sb.append(token);
        return sb.toString();
    }

    public static String findInArray(Object[] haystack, String needle) {
        for(Object item : haystack)
            if(item.toString().contains(needle))
                return item.toString();
        return null;
    }

    public static boolean equals(String s1, String s2) {
        if (s1==null && s2==null)
            return true;
        else if (s1==null || s2==null)
            return false;
        else
            return s1.equals(s2);
    }
}