from Bot import contextual as ctl

v = ctl.Bot()
v.organize_data('intents.json')
v.remove_duplicates()
v.trainer()
v.neural_network()

while True:
	i = input("Want to continue? ")
	if i == '1':
		n = input('your question ')
		v.response(n)
	else:
		break