package ix.complexity.lucene3.esa;

public interface RelatednessComparable {

    /** Return value, if no relatedness could be computed, because one or both terms are missing in Wikipedia. */
    public static final double NOT_IN_WIKIPEDIA = -1.0;
    
    /** Return value, if no relatedness could be computed, because one or both terms are not categorized. */
    public static final double NOT_CATEGORIZED = -2.0;

    /** Return value, if a disambiguation page was hit, but the disambiguation module could not find a suitable sense. */
    public static final double NO_SENSE = -3.0;

    /**
     * Computes the semantic relatedness between two terms.
     * The function should test whether the two tokens are actually in the semantic net and return -1 otherwise.
     * It will not indicate which of the two terms was not present. If this information is needed this should be checked separately. 
     * @param token1 The first term.
     * @param token2 The second term
     * @return The semantic relatedness value. [0,1]
     * 0 means no relatedness at all. 1 means full relatedness.
     * -1 means that no relatedness could have been computed.
     * @throws Exception  
     */
    public double getRelatedness(String token1, String token2) throws EsaException;

}
