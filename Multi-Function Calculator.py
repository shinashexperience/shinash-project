import math
from fractions import Fraction
import sympy

def solve_proportions():
    print("Solve proportions in the form: a/b = c/d")
    try:
        a = float(input("Enter a: "))
        b = float(input("Enter b: "))
        c = float(input("Enter c: "))
        d = input("Enter d (or 'x' for unknown): ")

        if d == 'x':
            if a == 0:
                print("Error: Division by zero")
            else:
                x = (b * c) / a
                print(f"x = {x}")
        else:
            d = float(d)
            if a * d == b * c:
                print("Proportion is true")
            else:
                print("Proportion is false")
    except ValueError:
        print("Invalid input")

def solve_linear_equation():
    print("Solve for x in equations like: ax + b = c")
    try:
        a = float(input("Enter a: "))
        b = float(input("Enter b: "))
        c = float(input("Enter c: "))
        
        if a == 0:
            if b == c:
                print("Infinite solutions")
            else:
                print("No solution")
        else:
            x = (c - b) / a
            print(f"x = {x}")
    except ValueError:
        print("Invalid input")

def simplify_square_root():
    print("Simplify square roots like √12 = 2√3")
    try:
        num = int(input("Enter number under root: "))
        if num < 0:
            print("Cannot simplify negative square roots")
            return
        
        largest_square = 1
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % (i*i) == 0:
                largest_square = i*i

        coefficient = int(math.sqrt(largest_square))
        remaining = num // largest_square

        if remaining == 1:
            print(f"√{num} = {coefficient}")
        elif coefficient == 1:
            print(f"√{num} = √{remaining}")
        else:
            print(f"√{num} = {coefficient}√{remaining}")
    except ValueError:
        print("Invalid input")

def decimal_to_fraction_percent():
    try:
        decimal = float(input("Enter decimal: "))
        fraction = Fraction(decimal).limit_denominator()
        percent = decimal * 100
        print(f"Fraction: {fraction}")
        print(f"Percent: {percent}%")
    except ValueError:
        print("Invalid input")

def fraction_to_decimal_percent():
    try:
        numerator = int(input("Enter numerator: "))
        denominator = int(input("Enter denominator: "))
        decimal = numerator / denominator
        percent = decimal * 100
        print(f"Decimal: {decimal}")
        print(f"Percent: {percent}%")
    except (ValueError, ZeroDivisionError):
        print("Invalid input")

def percent_to_decimal_fraction():
    try:
        percent = float(input("Enter percent: "))
        decimal = percent / 100
        fraction = Fraction(decimal).limit_denominator()
        print(f"Decimal: {decimal}")
        print(f"Fraction: {fraction}")
    except ValueError:
        print("Invalid input")

def main():
    while True:
        print("\n=== Multi-Function Calculator ===")
        print("1. Solve Proportions")
        print("2. Solve Linear Equations")
        print("3. Simplify Square Roots")
        print("4. Decimal to Fraction/Percent")
        print("5. Fraction to Decimal/Percent")
        print("6. Percent to Decimal/Fraction")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            solve_proportions()
        elif choice == '2':
            solve_linear_equation()
        elif choice == '3':
            simplify_square_root()
        elif choice == '4':
            decimal_to_fraction_percent()
        elif choice == '5':
            fraction_to_decimal_percent()
        elif choice == '6':
            percent_to_decimal_fraction()
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()