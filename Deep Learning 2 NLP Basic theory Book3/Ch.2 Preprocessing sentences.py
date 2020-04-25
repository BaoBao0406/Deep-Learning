import re

text = 'You say goodbye and I say hello.'

text = text.lower()
text = text.replace('.', ' .')
print(text)

words = text.split(' ')
print(words)

#words1 = re.split('(\W+)?', text)
#print(words1)

word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(id_to_word)
print(word_to_id)

print(id_to_word[1])
print(word_to_id['hello'])


import numpy as np
corpus = [word_to_id[w] for w in words]
print(corpus)
corpus = np.array(corpus)
print(corpus)