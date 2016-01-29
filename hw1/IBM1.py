#!/usr/bin/env python
import optparse
import sys
import time
from collections import defaultdict

class IBM1:
	def __init__(self, opts):
		self.bitext = self.read_corpus(opts)
		self.translation_probability, self.count = self.initialize_parameter()

	def read_corpus(self, opts):
		sys.stderr.write(">> Reading corpus...\n")
		bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
		return bitext

	def initialize_parameter(self):
		sys.stderr.write(">> Initializing model parameters...\n")
		translation_probability = defaultdict(float)
		count = defaultdict(int)
		for (n, (f, e)) in enumerate(self.bitext):
			for f_i in f:
				for e_j in e:
					translation_probability[(f_i, e_j)] = 1 / float(len(f))
			for e_j in e:
				count[e_j] = defaultdict(int)
			for e_j in e:
				for f_i in f:
					count[e_j][f_i] = 0
		return translation_probability, count

	def em_algorithm(self):
		sys.stderr.write(">> EM algorithm...\n")
		for iteration in range(opts.iteration_number):
			sys.stderr.write(" Iteration: %i" % (iteration + 1))
			for e_j in self.count.keys():
				for f_i in self.count[e_j].keys():
					self.count[e_j][f_i] = 0
			
			sys.stderr.write(" Calculating delta...\n")
			delta = defaultdict(float)
			for (n, (f, e)) in enumerate(self.bitext):
				for f_i in f:
					summation = sum([self.translation_probability[(f_i, e_j)] for e_j in e])
					for e_j in e:
						delta = self.translation_probability[(f_i, e_j)] / summation
						self.count[e_j][f_i] += delta

			sys.stderr.write(" Updating translation probability...\n")
			summation_dict = {}
			for (f_i, e_j) in self.translation_probability.keys():
				if e_j not in summation_dict:
					summation_dict[e_j] = sum(self.count[e_j].values())
				self.translation_probability[(f_i, e_j)] = self.count[e_j][f_i] / summation_dict[e_j]

	def predict_alignment(self):
		sys.stderr.write(" Predicting alignment...\n")
		for (f, e) in self.bitext:
			for (i, f_i) in enumerate(f):
				probabilities = [self.translation_probability[(f_i, e_j)] for e_j in e]
				pos = probabilities.index(max(probabilities))
				sys.stdout.write("%i-%i " % (i, pos))
			sys.stdout.write("\n")


if __name__ == "__main__":
	optparser = optparse.OptionParser()
	optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
	optparser.add_option("-i", "--iteration_number", dest="iteration_number", default=5, type="int", help="Number of iterations for EM algorithm (default=5)")
	optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
	(opts, _) = optparser.parse_args()

	t0=time.clock()

	ibm1 = IBM1(opts)
	ibm1.em_algorithm()
	ibm1.predict_alignment()

	sys.stdout.write("run time: %f sec \n" % (time.clock() - t0))

