import nltk
from enchant.checker import SpellChecker
from nltk.corpus import treebank
from nltk.corpus import wordnet as wn
from nltk.grammar import CFG, Nonterminal
from itertools import chain


# From: http://stackoverflow.com/questions/21207414/why-does-nltk-wordnet-fail-finding-simple-words
all_lemmas = set(chain(*[i.lemma_names for i in wn.all_synsets()]))
def in_wordnet(word):
  return True if word in all_lemmas else False


tbank_productions = set(production for sent in treebank.parsed_sents()
                        for production in sent.productions())
tbank_grammar = CFG(Nonterminal('S'), list(tbank_productions))

rd_parser = nltk.parse.EarleyChartParser(tbank_grammar)


paragraph = "The scale of the monitoring carried out by the NSA has been revealed in documents made public by whistleblower Edward Snowden over the last two years. Some of those papers show the NSA tapped the net's backbone network to siphon off data. The backbone is made up of high-speed cables that link big ISPs and key transit points on the net."


chkr = SpellChecker("en_US", paragraph)
for err in chkr:
	paragraph = paragraph.replace(err,"spam")

sent = paragraph.split()

for word in sent:
	if not wordnet.is_known(word) and not d.check(word):
		# Correct this shit
		print(d.suggest(word))
		paragraph = paragraph.replace(word,d.suggest(word)[0])
		print("Word: " + word)
		print("Replace With: " + d.suggest(word)[0])
		print("New Paragraph: " + paragraph)

sent = paragraph.split()

#for word in sent:
#	print(word)

for tree in rd_parser.parse(sent):
	print(tree)

# mini_grammar = ContextFreeGrammar(Nonterminal('S'), treebank.parsed_sents()[0].productions())
# parser = nltk.parse.EarleyChartParser(mini_grammar)
# print parser.parse(treebank.sents()[0])


