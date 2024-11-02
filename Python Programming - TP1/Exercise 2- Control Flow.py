# Exercice 2

# a - enter a num
number = int(input("Enter a number: "))

# b - divided by 3
# c - divided by 5
# d - divided by 5 and 5
# e - print num otherwise

if number % 3 == 0 and number % 5 == 0 :
    print("FizzBuzz")
elif number % 3 == 0 :
    print("Fizz")
elif number % 5 == 0 :
    print("Buzz")
else :
    print(number)