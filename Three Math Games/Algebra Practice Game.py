import random

def algebra_game():
    print("=== Algebra Practice Game ===")
    
    difficulty = input("Pilih kesulitan (mudah/sulit): ").lower()
    
    if difficulty == "sulit":
        num_range = 100
        problems = 10
    else:
        num_range = 20
        problems = 5
    
    score = 0
    
    for i in range(problems):
        print(f"\nSoal {i+1}:")
        
        # Randomly choose problem type (1-step or 2-step)
        problem_type = random.choice(['one_step', 'two_step'])
        
        if problem_type == 'one_step':
            # ax + b = c
            a = random.randint(1, num_range//10)
            b = random.randint(-num_range, num_range)
            c = random.randint(-num_range, num_range)
            
            # Calculate answer: x = (c - b) / a
            answer = (c - b) / a
            
            # Make sure answer is integer for simplicity
            if not answer.is_integer():
                a = 1  # Simplify to x + b = c
            
            answer = int((c - b) / a)
            equation = f"{a if a != 1 else ''}x {'+' if b >= 0 else ''}{b} = {c}"
            
        else:  # two_step
            # a(bx + c) = d
            a = random.randint(2, 5)
            b = random.randint(1, 5)
            c = random.randint(-num_range//2, num_range//2)
            d = random.randint(-num_range, num_range)
            
            # Calculate answer: x = (d/a - c) / b
            answer = (d/a - c) / b
            
            if not answer.is_integer():
                # Adjust to make integer answer
                d = a * (b * random.randint(-num_range//10, num_range//10) + c)
            
            answer = int((d/a - c) / b)
            equation = f"{a}({b}x {'+' if c >= 0 else ''}{c}) = {d}"
        
        print(f"Selesaikan: {equation}")
        
        try:
            user_answer = int(input("x = "))
            
            if user_answer == answer:
                print("✅ Benar!")
                score += 1
            else:
                print(f"❌ Salah! Jawaban yang benar: {answer}")
        
        except ValueError:
            print("❌ Masukkan harus berupa angka!")
    
    print(f"\nSkor akhir: {score}/{problems} ({score/problems*100:.1f}%)")

# Test the game
algebra_game()