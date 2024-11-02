# Exercise 4

# a - positive int as input
def calc_factorial(n) :
    # d - error hundling
    if not isinstance(n, int) :
        raise ValueError("Input should be an integer")
    if n < 0 :
        raise ValueError("Input should be poitive number")
    
    # b - factorial of that num
    # c - returning the results
    if n == 0 or n == 1 :
        return 1
    else :
        return n * calc_factorial(n - 1)

# e - main function that call the other func above to calc factorial
def main() :
    try :
        num = int(input("Enter a positive integer to calculate its factorial: "))
        res = calc_factorial(num)
        print(f"The factorial of {num} is: {res}")
    except ValueError as e :
        print(f"Error: {e}")

if __name__ == "__main__" :
    main()