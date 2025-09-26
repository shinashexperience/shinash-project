import random
import matplotlib.pyplot as plt

def scatter_plot_game():
    print("=== Scatter Plot Game ===")
    print("Tebak koordinat (x,y) dari titik yang ditandai!")
    
    # Generate random points
    num_points = random.randint(3, 8)
    points = []
    
    # Determine graph size based on difficulty
    difficulty = input("Pilih kesulitan (mudah/sulit): ").lower()
    if difficulty == "sulit":
        max_range = 50
    else:
        max_range = 20
    
    for _ in range(num_points):
        x = random.randint(-max_range, max_range)
        y = random.randint(-max_range, max_range)
        points.append((x, y))
    
    # Select target point
    target_point = random.choice(points)
    
    # Plot points
    plt.figure(figsize=(8, 6))
    x_vals, y_vals = zip(*points)
    plt.scatter(x_vals, y_vals, color='blue', s=100)
    plt.scatter([target_point[0]], [target_point[1]], color='red', s=200, marker='X')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.title(f"Scatter Plot Game - Tebak titik merah (Range: ±{max_range})")
    
    plt.show()
    
    # Get user input
    try:
        guess_x = int(input("Masukkan koordinat x: "))
        guess_y = int(input("Masukkan koordinat y: "))
        
        if guess_x == target_point[0] and guess_y == target_point[1]:
            print("✅ Benar! Jawaban tepat!")
        else:
            print(f"❌ Salah! Jawaban yang benar: ({target_point[0]}, {target_point[1]})")
    
    except ValueError:
        print("❌ Masukkan harus berupa angka!")

# Test the game
scatter_plot_game()