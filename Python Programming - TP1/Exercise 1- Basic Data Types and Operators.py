# Exercice 1

# a - enter two numbers
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))

# b - opearations
add = f"addition : {num1} + {num2} = {num1 + num2}"
sub = f"subtraction : {num1} + {num2} = {num1 - num2}"
mul = f"multiplication : {num1} + {num2} = {num1 * num2}"
if (num2 != 0) :
    div = f"division : {num1} + {num2} = {num1 / num2}"
else :
    div = f"division is not possible - div by zero"

# c - printing operation
print(add,"\n")
print(sub,"\n")
print(mul,"\n")
print(div)

# d - greater than ,less or equal and print res
if num1 > num2:
    print(f"{num1} is greater than {num2}")
elif num1 < num2:
    print(f"{num1} is less than {num2}")
else :
    print(f"{num1} is equal to {num2}")


