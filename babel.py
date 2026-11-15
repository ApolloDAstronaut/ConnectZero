### Unrelated to the project
### prints all books in the Library of Babel

pages = 1
lines = 1
letters = 10
alphabet = 'abcdefghijklmnopqrstuvwxyz!?",.:;'

len_alpha = len(alphabet)
len_book = pages * lines * letters
count_thingy = [0] * len_book

for i in range(len_book ** len_alpha):
    print(''.join([alphabet[count_thingy[j]] for j in range(len_book)]))
    count_thingy[0] += 1
    for j in range(len(count_thingy)):
        if count_thingy[j] >= len_alpha:
            if j + 1 > len(count_thingy):
                break
            count_thingy[j] = 0
            count_thingy[j + 1] += 1


