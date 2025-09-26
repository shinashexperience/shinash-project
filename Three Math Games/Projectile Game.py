import random
import matplotlib.pyplot as plt
import numpy as np

def projectile_game():
    print("=== Projectile Game ===")
    print("Atur parabola untuk melewati tembok!")
    
    level = input("Pilih level (1 untuk slider, 2 untuk input manual): ")
    
    # Generate random wall
    wall_x = random.randint(3, 8)
    wall_height = random.uniform(1, 5)
    wall_width = 0.5
    
    # Parabola parameters
    if level == '1':
        print("\nGunakan slider untuk mengatur a, b, c")
        print(f"Tembok: tinggi {wall_height:.1f} pada x = {wall_x}")
        
        # For slider version, we'll use interactive plot
        from matplotlib.widgets import Slider
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.3)
        
        # Initial parameters
        a0, b0, c0 = -0.5, 5, 0
        
        # Create sliders
        ax_a = plt.axes([0.2, 0.2, 0.6, 0.03])
        ax_b = plt.axes([0.2, 0.15, 0.6, 0.03])
        ax_c = plt.axes([0.2, 0.1, 0.6, 0.03])
        
        slider_a = Slider(ax_a, 'a', -2.0, 0.0, valinit=a0)
        slider_b = Slider(ax_b, 'b', 0.0, 10.0, valinit=b0)
        slider_c = Slider(ax_c, 'c', -5.0, 5.0, valinit=c0)
        
        def update(val):
            ax.clear()
            a = slider_a.val
            b = slider_b.val
            c = slider_c.val
            
            # Plot parabola
            x = np.linspace(0, 10, 100)
            y = a*x**2 + b*x + c
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Plot wall
            ax.plot([wall_x, wall_x], [0, wall_height], 'r-', linewidth=10)
            ax.fill_between([wall_x-wall_width/2, wall_x+wall_width/2], 
                           [wall_height, wall_height], color='red', alpha=0.5)
            
            # Check if clears wall
            y_at_wall = a*wall_x**2 + b*wall_x + c
            if y_at_wall > wall_height:
                status = "✅ CLEAR!"
                color = 'green'
            else:
                status = "❌ HIT!"
                color = 'red'
            
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Projectile Game - {status}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.canvas.draw_idle()
        
        slider_a.on_changed(update)
        slider_b.on_changed(update)
        slider_c.on_changed(update)
        
        # Initial plot
        update(None)
        plt.show()
        
    else:  # Level 2 - manual input
        print(f"\nTembok: tinggi {wall_height:.1f} pada x = {wall_x}")
        print("Masukkan parameter parabola y = ax² + bx + c")
        
        try:
            a = float(input("a = "))
            b = float(input("b = "))
            c = float(input("c = "))
            
            # Calculate height at wall
            y_at_wall = a*wall_x**2 + b*wall_x + c
            
            # Plot result
            x = np.linspace(0, 10, 100)
            y = a*x**2 + b*x + c
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, 'b-', linewidth=2, label=f'y = {a}x² + {b}x + {c}')
            
            # Plot wall
            plt.plot([wall_x, wall_x], [0, wall_height], 'r-', linewidth=10, label='Tembok')
            plt.fill_between([wall_x-wall_width/2, wall_x+wall_width/2], 
                           [wall_height, wall_height], color='red', alpha=0.5)
            
            # Mark the point at wall
            plt.plot(wall_x, y_at_wall, 'go', markersize=8, label=f'Ketinggian di tembok: {y_at_wall:.2f}')
            
            if y_at_wall > wall_height:
                status = "✅ BERHASIL! Parabola melewati tembok!"
                plt.title(f'Projectile Game - {status}', color='green')
            else:
                status = f"❌ GAGAL! Kurang {wall_height - y_at_wall:.2f} unit"
                plt.title(f'Projectile Game - {status}', color='red')
            
            plt.xlim(0, 10)
            plt.ylim(0, max(10, y_at_wall + 2))
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            
            print(status)
            
        except ValueError:
            print("❌ Masukkan harus berupa angka!")

# Test the game
projectile_game()