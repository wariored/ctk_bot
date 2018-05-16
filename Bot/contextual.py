
class Bot:
	"""This is for creating Bot. It will help you to build conversational model"""
	def __init__(self):
		self.intents = {}
		self.words = []
		self.classes = []
		self.documents = []
		self.ignore_words = ['?']
		self.train_x = []
		self.train_y = []
		self.model = None

	def add_ignore_words(self, ignore_words):
		"""Add some words to ignore during the processus"""
		[self.ignore_words.append(w) for w in ignore_words if w not in self.ignore_words]

	def organize_data(self, json_file):
		"""Load your json file and organize them in words, classes and documents
		Anytime you call this function, you should call remove_duclipcates function too"""
		import nltk
		import json
		with open(json_file) as json_data:
			intents = json.load(json_data)
			# loop through each sentence in our intents patterns
			for intent in intents['intents']:
				for pattern in intent['patterns']:
					# tokenize each word in the sentence
					w = nltk.word_tokenize(pattern)
					# add to our words list
					self.words.extend(w)
					# add to documents in our corpus
					self.documents.append((w, intent['tag']))
				if intent['tag'] not in self.classes:
					self.classes.append(intent['tag'])
			self.intents = intents

	def remove_duplicates(self):
		"""stem, lower each word and remove duplicate and sort words and classes"""
		from nltk.stem.lancaster import LancasterStemmer
		stemmer = LancasterStemmer()
		# stem and lower each word and remove duplicates
		words = [stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
		self.words = sorted(list(set(words)))
		# remove duplicates
		self.classes = sorted(list(set(self.classes)))
	def trainer(self):
		from Bot.training import train
		self.train_x, self.train_y = train(self.documents, self.words, self.classes)
	def neural_network(self, fitting=False, n_epoch=1000, batch_size=8):
		from Bot.model import tensorflow_model
		self.model = tensorflow_model(self.train_x, self.train_y, n_epoch, batch_size, fitting)

	def classify(self, sentence, show_details=False):
		"""Classifier for sentence received from user"""
		import Bot.training as training
		# generate probabilities from the model
		results = self.model.predict([training.bow(sentence, self.words, show_details)])[0]
		# filter out predictions below a threshold
		results = [[i,r] for i,r in enumerate(results) if r > training.ERROR_THRESHOLD]
		# sort by strength of probability
		results.sort(key=lambda x: x[1], reverse=True)
		return_list = []
		for r in results:
			return_list.append((self.classes[r[0]], r[1]))
		# return tuple of intent and probability
		return return_list

	def response(self, sentence, userID='123', show_details=False):
		"""Response of the chat"""
		import random
		results = self.classify(sentence)
		# if we have a classification then find the matching intent tag
		if results:
			# loop as long as there are matches to process
			while results:
				for i in self.intents['intents']:
					# find a tag matching the first result
					if i['tag'] == results[0][0]:
						# a random response from the intent
						return print(random.choice(i['responses']))
				results.pop(0)
