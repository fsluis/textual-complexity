This repository contains the main Scala code for text analysis.

## Contents
The Scala code is divided into subprojects according to shared dependencies and functionality.

The following subprojects are relevant:
* __data__. The main control flow for reading data sets, analysing features, and submitting the results back to a database. Complexity features are derived in the scala package _ix/data/com/features_. This is the recommended starting point for review.
* __complexity-*__. Specifies different _Services_ and routines that calculate aspects of textual complexity. For example, the __complexity-lm__ subproject handles the interaction with KenLM and AltLex for calculating the respective language model and AltLex features.

The following subprojects can be considered boilerplate:
* __integration__ merges all subprojects into a single dependency.  
* __common-*__ contains shared routines and boilerplate used by the different subprojects.
* __spark-extkey__ combines all test code into a fat-jar used for testing.
* __shaded__ contains external projects with specific dependencies dropped or changed, to solve dependency conflicts. 

## Resources
The Scala code relies on extensive resources to run. It needs:
* Input data. Currently included is a file with 10 pairs of simple and English Wikipedia articles (at resources/data) for testing.
* WordNet. Currently included is version 3.1.
* A large language model (CommonCrawl). Can be obtained from http://statmt.org/ngrams.
* Hyphen dictionaires. Currently included in resources/hyphen 
* ESA model. Can be generated using data/src/scala/ix/complexity/lucene3/esa/index/
