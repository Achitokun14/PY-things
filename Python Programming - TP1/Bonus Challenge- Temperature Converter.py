# Bonus challenge

# a - Defines two functions
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def main():
    # b - Presents a menu to the user to choose which conversion they want to perform
    while True:
        print("\nTemperature Converter")
        print("1. Celsius to Fahrenheit")
        print("2. Fahrenheit to Celsius")
        print("3. Quit")
        
        choice = input("Enter your choice (1-3): ")
        
        # e - Allows the user to perform multiple conversions until they choose to quit
        if choice == '3':
            print("Goodbye!")
            break
        
        # c - Asks for the temperature value and performs the conversion 
        # d - Prints the result rounded to two decimal places
        try:
            if choice == '1':
                celsius = float(input("Enter temperature in Celsius: "))
                fahrenheit = celsius_to_fahrenheit(celsius)
                print(f"{celsius}°C = {fahrenheit:.2f}°F")
            elif choice == '2':
                fahrenheit = float(input("Enter temperature in Fahrenheit: "))
                celsius = fahrenheit_to_celsius(fahrenheit)
                print(f"{fahrenheit}°F = {celsius:.2f}°C")
            else:
                print("Invalid choice! Please try again.")
        except ValueError:
            print("Invalid input! Please enter a number.")

if __name__ == "__main__":
    main()