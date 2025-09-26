import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import symbols, solve, factor
import math

class GraphingCalculator:
    def __init__(self):
        self.x = symbols('x')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.x_min, self.x_max = -10, 10
        self.y_min, self.y_max = -10, 10
        
    def graph_function(self):
        """Graph one or more functions"""
        try:
            self.ax.clear()
            functions = input("Enter function(s) separated by commas (e.g., 'x**2', '2*x+1'): ").split(',')
            
            x_vals = np.linspace(self.x_min, self.x_max, 400)
            
            for i, func_str in enumerate(functions):
                func_str = func_str.strip()
                # Convert string to lambda function
                y_vals = eval(func_str.replace('^', '**'), {'x': x_vals, 'np': np, 'math': math})
                
                self.ax.plot(x_vals, y_vals, label=f'y = {func_str}', linewidth=2)
            
            self.setup_graph()
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")

    def create_table(self):
        """Create a table of (x,y) values"""
        try:
            func_str = input("Enter function (e.g., 'x**2'): ").strip()
            start = float(input("Start x: "))
            end = float(input("End x: "))
            step = float(input("Step size: "))
            
            print(f"\nTable for y = {func_str}")
            print("x\t|\ty")
            print("-" * 20)
            
            x_val = start
            while x_val <= end:
                y_val = eval(func_str.replace('^', '**'), {'x': x_val, 'np': np, 'math': math})
                print(f"{x_val:.2f}\t|\t{y_val:.2f}")
                x_val += step
                
        except Exception as e:
            print(f"Error: {e}")

    def shade_above_below(self):
        """Shade above or below the line"""
        try:
            self.ax.clear()
            func_str = input("Enter function (e.g., '2*x+1'): ").strip()
            shade_direction = input("Shade (a)bove or (b)elow? ").lower()
            
            x_vals = np.linspace(self.x_min, self.x_max, 400)
            y_vals = eval(func_str.replace('^', '**'), {'x': x_vals, 'np': np, 'math': math})
            
            self.ax.plot(x_vals, y_vals, 'b-', label=f'y = {func_str}', linewidth=2)
            
            if shade_direction == 'a':
                self.ax.fill_between(x_vals, y_vals, self.y_max, alpha=0.3, color='red')
                shade_label = "Above line"
            else:
                self.ax.fill_between(x_vals, y_vals, self.y_min, alpha=0.3, color='blue')
                shade_label = "Below line"
            
            self.ax.legend([f'y = {func_str}', shade_label])
            self.setup_graph()
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")

    def solve_system(self):
        """Solve and graph a system of equations"""
        try:
            self.ax.clear()
            eq1 = input("Enter first equation (e.g., '2*x + 1'): ").strip()
            eq2 = input("Enter second equation (e.g., '-x + 4'): ").strip()
            
            # Solve symbolically
            solution = solve([sp.sympify(eq1.replace('^', '**')) - sp.sympify(eq2.replace('^', '**'))], self.x)
            x_sol = solution[self.x]
            y_sol = eval(eq1.replace('^', '**'), {'x': x_sol, 'np': np, 'math': math})
            
            print(f"Solution: x = {x_sol:.2f}, y = {y_sol:.2f}")
            
            # Graph both equations
            x_vals = np.linspace(self.x_min, self.x_max, 400)
            y1_vals = eval(eq1.replace('^', '**'), {'x': x_vals, 'np': np, 'math': math})
            y2_vals = eval(eq2.replace('^', '**'), {'x': x_vals, 'np': np, 'math': math})
            
            self.ax.plot(x_vals, y1_vals, 'b-', label=f'y = {eq1}', linewidth=2)
            self.ax.plot(x_vals, y2_vals, 'r-', label=f'y = {eq2}', linewidth=2)
            self.ax.plot(x_sol, y_sol, 'go', markersize=8, label=f'Solution ({x_sol:.2f}, {y_sol:.2f})')
            
            self.setup_graph()
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")

    def zoom_graph(self):
        """Zoom in or out on a graph"""
        try:
            zoom_direction = input("Zoom (i)n or (o)ut? ").lower()
            factor = float(input("Zoom factor (e.g., 2 for 2x): "))
            
            if zoom_direction == 'i':
                # Zoom in
                x_range = self.x_max - self.x_min
                y_range = self.y_max - self.y_min
                self.x_min += x_range / (2 * factor)
                self.x_max -= x_range / (2 * factor)
                self.y_min += y_range / (2 * factor)
                self.y_max -= y_range / (2 * factor)
            else:
                # Zoom out
                x_range = self.x_max - self.x_min
                y_range = self.y_max - self.y_min
                self.x_min -= x_range / (2 * factor)
                self.x_max += x_range / (2 * factor)
                self.y_min -= y_range / (2 * factor)
                self.y_max += y_range / (2 * factor)
            
            print(f"New view: x=[{self.x_min:.1f}, {self.x_max:.1f}], y=[{self.y_min:.1f}, {self.y_max:.1f}]")
            
        except Exception as e:
            print(f"Error: {e}")

    def solve_quadratic(self):
        """Solve quadratic equations"""
        try:
            a = float(input("Enter a: "))
            b = float(input("Enter b: "))
            c = float(input("Enter c: "))
            
            # Calculate discriminant
            discriminant = b**2 - 4*a*c
            
            if discriminant > 0:
                root1 = (-b + math.sqrt(discriminant)) / (2*a)
                root2 = (-b - math.sqrt(discriminant)) / (2*a)
                print(f"Two real roots: x = {root1:.2f}, x = {root2:.2f}")
            elif discriminant == 0:
                root = -b / (2*a)
                print(f"One real root: x = {root:.2f}")
            else:
                real_part = -b / (2*a)
                imaginary_part = math.sqrt(-discriminant) / (2*a)
                print(f"Two complex roots: {real_part:.2f} ± {imaginary_part:.2f}i")
            
            # Graph the quadratic function
            self.ax.clear()
            x_vals = np.linspace(self.x_min, self.x_max, 400)
            y_vals = a*x_vals**2 + b*x_vals + c
            
            self.ax.plot(x_vals, y_vals, 'purple', label=f'y = {a}x² + {b}x + {c}', linewidth=2)
            self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Plot roots if they are real
            if discriminant >= 0:
                if discriminant > 0:
                    roots = [root1, root2]
                else:
                    roots = [root]
                
                for root in roots:
                    if self.x_min <= root <= self.x_max:
                        y_root = a*root**2 + b*root + c
                        self.ax.plot(root, y_root, 'ro', markersize=8)
            
            self.setup_graph()
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")

    def setup_graph(self):
        """Setup graph appearance"""
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Graphing Calculator')
        self.ax.legend()

    def reset_view(self):
        """Reset zoom to default"""
        self.x_min, self.x_max = -10, 10
        self.y_min, self.y_max = -10, 10
        print("View reset to default")

    def main_menu(self):
        """Main menu interface"""
        while True:
            print("\n" + "="*50)
            print("           GRAPHING CALCULATOR")
            print("="*50)
            print("1. Graph one or more functions")
            print("2. Create a table of (x,y) values")
            print("3. Shade above or below the line")
            print("4. Solve and graph a system of equations")
            print("5. Zoom in or out on a graph")
            print("6. Solve quadratic equations")
            print("7. Reset zoom view")
            print("8. Exit")
            print("="*50)
            
            choice = input("Enter your choice (1-8): ")
            
            if choice == '1':
                self.graph_function()
            elif choice == '2':
                self.create_table()
            elif choice == '3':
                self.shade_above_below()
            elif choice == '4':
                self.solve_system()
            elif choice == '5':
                self.zoom_graph()
            elif choice == '6':
                self.solve_quadratic()
            elif choice == '7':
                self.reset_view()
            elif choice == '8':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

# Run the calculator
if __name__ == "__main__":
    calculator = GraphingCalculator()
    calculator.main_menu()