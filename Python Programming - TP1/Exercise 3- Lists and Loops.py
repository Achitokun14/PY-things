# Exercice 3

import random

# a - 10 random nums between 1 and 100
numbers = [random.randint(1, 100) for _ in range(10)]

# b - print original list
print("Original list: ", numbers)

# c - sum of all numbers
total = 0
for num in numbers:
    total += num
print("Sum of numbers: ", total)

# d - max and min val
print("Max value: ", max(numbers))
print("Max value: ", min(numbers))

# e - even nums from original list
even_nums = [num for num in numbers if num % 2 == 0]
print("Even numbers: ", even_nums)
