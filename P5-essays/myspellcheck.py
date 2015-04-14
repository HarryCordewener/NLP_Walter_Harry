##
## Sources found in here: http://stackoverflow.com/questions/22073688/python-spell-corrector-using-ntlk
## 

import nltk
import enchant
from enchant.checker import SpellChecker
from nltk.corpus import treebank
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import wordnet as wn
from nltk.grammar import CFG, Nonterminal
from nltk.metrics.distance import edit_distance
from itertools import chain


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


if __name__ == '__main__':
    filename = "C:/Users/Harry/OneDrive/P5-essays/P5/P5-original/high/11580.txt"
    f = open(filename, 'r')
    text = "The scale of the monitoring carried out by the NSA has been revealed in documents made public by whistleblower Edward Snowden over the last two years. Some of those papers show the NSA tapped the net's backbone network to siphon off data. The backbone is made up of high-speed cables that link big ISPs and key transit points on the net."
    text = f.read()    

    my_spell_checker = MySpellChecker(max_dist=1)
    chkr = SpellChecker("en_US", text)
    for err in chkr:
        print(err.word + " at position " + str(err.wordpos))
        err.replace(my_spell_checker.replace(err.word))


    t = chkr.get_text()
    print("\n" + t)


    tbank_productions = set(production for sent in treebank.parsed_sents()
                            for production in sent.productions())
    tbank_grammar = CFG(Nonterminal('S'), list(tbank_productions))

    rd_parser = nltk.parse.EarleyChartParser(tbank_grammar)
    
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    print('\n-----\n'.join(sent_detector.tokenize(t.strip())))

    print(TreebankWordTokenizer().tokenize(t))


    ##sent = text.split()

    ##for tree in rd_parser.parse(sent):
      ##  print(tree)



