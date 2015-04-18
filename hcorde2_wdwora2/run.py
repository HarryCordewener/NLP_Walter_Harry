##
## Sources found in here: http://stackoverflow.com/questions/22073688/python-spell-corrector-using-ntlk
##

import sys
import os
import nltk
import enchant
import json
from enchant.checker import SpellChecker
from nltk import *
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from nltk.grammar import CFG, Nonterminal, DependencyGrammar
from nltk.metrics.distance import edit_distance
from itertools import chain


statistics = dict()

class MySpellChecker():
    def __init__(self, dict_name='en_US', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        suggestions = self.spell_dict.suggest(word)

        if suggestions:
            for suggestion in suggestions:
                if edit_distance(word, suggestion) <= self.max_dist:
                    return suggestions[0]

        return word


def train(f, level):
    truelevel = ""
    if level.find("high") != -1: truelevel = "high"
    if level.find("low") != -1: truelevel = "low"
    if level.find("medium") != -1: truelevel = "medium"
    if truelevel == "": return # We are not looking at a proper file
    if str(f).find(".txt") == -1: return # We are not looking at a proper file
    print("This " + truelevel + " level file is good to go.")

    text = f.read() 

    spellerrors = 0
    
    my_spell_checker = MySpellChecker(max_dist=1)
    chkr = SpellChecker("en_US", text)
    for err in chkr:
        # print(err.word + " at position " + str(err.wordpos))
        err.replace(my_spell_checker.replace(err.word))
        spellerrors = spellerrors + 1;
    
    if( spellerrors < statistics.get(truelevel+"_error_min",pow(2,31)) ):
        statistics[truelevel+"_error_min"] = spellerrors
    if( spellerrors > statistics.get(truelevel+"_error_max",0) ):
        statistics[truelevel+"_error_max"] = spellerrors
    statistics[truelevel+"_error_total"] = statistics.get(truelevel+"_error_total",0) + spellerrors

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentencearray = sent_detector.tokenize(text.strip())
    sentencecount = len(sentencearray)

    if( sentencecount < statistics.get(truelevel+"_sentence_min",pow(2,31)) ):
        statistics[truelevel+"_sentence_min"] = sentencecount
    if( sentencecount > statistics.get(truelevel+"_sentence_max",0) ):
        statistics[truelevel+"_sentence_max"] = sentencecount
    statistics[truelevel+"_sentence_total"] = statistics.get(truelevel+"_sentence_total",0) + spellerrors

    statistics[truelevel+"_docs_total"] = statistics.get(truelevel+"_docs_total",0) + 1
    
    return

def checker(f, outf, thefilename):
    if str(f).find(".txt") == -1: return # We are not looking at a proper file
    print("Reading: " + str(f))
    text = f.read() 

    subverbagg_err = 0
    spellerrors = 0

    my_spell_checker = MySpellChecker(max_dist=1)
    chkr = SpellChecker("en_US", text)
    for err in chkr:
        print(err.word + " at position " + str(err.wordpos))
        err.replace(my_spell_checker.replace(err.word))
        spellerrors = spellerrors + 1;        

    t = chkr.get_text()
    # print("\n" + t)

    ## Useless code?
    #print("Trying to print semantics")
    #test_results = nltk.sem.util.interpret_sents(t, 'grammars/large_grammars/commandtalk.cfg')
    #for result in test_results:
    #    for (synrep, semrep) in result:
    #        print(synrep)

    ## This divides it into a per-sentence item list.
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentencearray = sent_detector.tokenize(t.strip())
    # print('\n-----\n'.join(sentencearray))

    ## This splits everything up into seperate words.
    tokenized = TreebankWordTokenizer().tokenize(t)
    # print(tokenized)

    ## Display spell errors found.
    print("Spellerrors: " + str(spellerrors))
    print("Words: " + str(len(tokenized)))
    print("Sentences: " + str(len(sentencearray)))

    for sentence in sentencearray:
        tokenized_sentence = TreebankWordTokenizer().tokenize(sentence)
        pos_tagged_sentence = nltk.pos_tag(tokenized_sentence)
        ## Illegal Combinations: http://grammar.ccc.commnet.edu/grammar/sv_agr.htm
        ## Basic Principle: Singular subjects need singular verbs; plural subjects need plural verbs.
        ## My brother is a nutritionist. My sisters are mathematicians.
        ## So we are counting errors for:
        ##  VBP / VBZ followed by NNS / NNPS
        ## We can't check for verb plural using the tags existing.
        for x in range(1,len(pos_tagged_sentence)):
            # print(str(pos_tagged_sentence[x-1]) + " & " + str(pos_tagged_sentence[x]))
            if( (pos_tagged_sentence[x-1][1] == "VBP" or pos_tagged_sentence[x-1][1] == "VBZ")
                and (pos_tagged_sentence[x][1] == "NNS" or pos_tagged_sentence[x][1] == "NNPS") ):
                # print("\nHOW DARE YOU!!!!\n")
                subverbagg_err = subverbagg_err + 1
    
    ## NEEDED: A dependency grammar!
    #pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
    #trees = pdp.parse(tokenized)
    #for tree in trees:
    #    print(tree)

    ## Work from here must be done using logic found at: http://www.nltk.org/book/ch05.html ?
    ## Note that the subject is: Do you agree or disagree with the following statement?
    ## "In twenty years, there will be fewer cars in use than there are today." Use reasons and examples to support your answer.
    ## 'cars', 'fewer', should be in there. Along with perhaps 'twenty' or '20'

    ## Grades, 1 = low, 5 = high
    ## 1a = Spelling Mistakes
    ## high_error_max -> high_error_min should give 3.5-5 points
    ## medium_error_max -> high_error_min should give 2.5-3.0 points
    ## low_error_max -> medium_error_min should give 1-2.5 points
    if(spellerrors > statistics["medium_error_max"] ):
        print("Higher than medium_error_max")
        score_1a = min(int(round(0.5+((1.5/(statistics["low_error_max"] - statistics["medium_error_max"])) * spellerrors))),1)
    elif(spellerrors > statistics["high_error_max"] ):
        print("Higher than high_error_max")
        score_1a = int(round(2+((1.5/(statistics["medium_error_max"] - statistics["high_error_max"])) * spellerrors)))
    else:
        print("Higher than high_error_min")
        print(3.5+((1.5/(statistics["high_error_max"] - statistics["high_error_min"])) * spellerrors))
        score_1a = max(int(round(3.5+((1.5/(statistics["high_error_max"] - statistics["high_error_min"])) * spellerrors))),5)
    
    # print(score_1a)

    ## 1b = Subject-Verb agreement - agreement with respect to person and number (singular/plural)
    score_1b = max(5-(subverbagg_err),1)

    ## 1c = Verb tense / missing verb / extra verb - is verb tense used correctly? Is a verb missing,
    ## e.g. an auxiliary? For example, in the example of low essay above, the sequence will
    ## be not agree is incorrect. Normally the verb to be is not followed by another infinitival
    ## verb, but either a participle or a progressive tense.
    score_1c = 0

    ## 1d =  Sentence formation - are the sentences formed properly? i.e. beginning and ending
    ## properly, is the word order correct, are the constituents formed properly? are there
    ## missing words or constituents (prepositions, subject, object etc.)?

    score_1d = 0

    ## 2. Semantics (meaning) / Pragmatics (general quality)
    ## (a) Is the essay coherent? Does it make sense?
    score_2a = 0

    ## (b) Does the essay address the topic?
    score_2b = 0
    #if "twenty" in t: score_2b = score_2b + 1
    #if "car" in t: score_2b = score_2b + 1
    #if "fewer" in t: score_2b = score_2b + 1
    #if "fewer cars" in t: score_2b = score_2b + 1
    # print(score_2b)

    ## 3. Length of the essay:
    ## (a) Is the length appropriate? At least 10 sentences were required. Longer essays are in
    ## general considered better.

    sentencenum = len(sentencearray)
    if(sentencenum > statistics["medium_sentence_max"] ):
        print("Higher than medium_sentence_max")
        score_3a = min(int(round(0.5+((1.5/(statistics["low_sentence_max"] - statistics["medium_sentence_max"])) * sentencenum))),1)
    elif(sentencenum > statistics["high_sentence_max"] ):
        print("Higher than high_sentence_max")
        score_3a = int(round(2+((1.5/(statistics["medium_sentence_max"] - statistics["high_sentence_max"])) * sentencenum)))
    else:
        print("Higher than high_sentence_min")
        print(3.5+((1.5/(statistics["high_sentence_max"] - statistics["high_sentence_min"])) * spellerrors))
        score_3a = max(int(round(3.5+((1.5/(statistics["high_sentence_max"] - statistics["high_sentence_min"])) * sentencenum))),5)

    
    # score_3a = min(max(int(1+((len(sentencearray)-10)/2)),1),5)
    # print(score_3a)

    ## Final Score = 1a + 1b + 1c + 2 ∗ 1d + 2 ∗ 2a + 3 ∗ 2b + 2 ∗ 3a
    score_final = score_1a + score_1b + score_1c + 2*score_1d + 2*score_2a + 3*score_2b + 2*score_3a
    output = (thefilename + "\t" + str(score_1a) + "\t" + str(score_1b) + "\t" + str(score_1c) + "\t" + str(score_1d) + 
             "\t" + str(score_2a) + "\t" + str(score_2b) + "\t" + str(score_3a) + "\t" + str(score_final) + "\tunknown")
    
    print(output)

    outf.write(output + "\n")

    return


if __name__ == '__main__':

    statfilename = os.path.join('input','training','trained_statistics.txt')
    outputfile = open(os.path.join('output','output.txt'),'w')
    
    if os.path.isfile(statfilename) != True:
        for subdir, dirs, files in os.walk(os.path.join('input','training','original')):
            for file in files:
                filename = os.path.join(subdir, file)
                f = open(filename, 'r')
                train(f,subdir)
                f.close
        with open(statfilename,'w') as outfile:
            json.dump(statistics, outfile)
    else:
        json1_file = open(statfilename,'r')
        statistics = json.load(json1_file)

    print(statistics)
    
    for subdir, dirs, files in os.walk(os.path.join('input','test')):
        for file in files:
            filename = os.path.join(subdir, file)
            print(filename)
            f = open(filename, 'r')
            checker(f, outputfile, file)
            f.close

    outputfile.close()
        
    
