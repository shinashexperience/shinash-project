import random
from collections import Counter

### Player Function ###
def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    if len(opponent_history) < 5:
        return random.choice(["R", "P", "S"])

    last_5 = opponent_history[-5:]
    move_counts = Counter(last_5)
    predicted = move_counts.most_common(1)[0][0]

    counter_moves = {"R": "P", "P": "S", "S": "R"}
    return counter_moves[predicted]

### Opponent Bots ###
def quincy(prev_play, _=[]):
    return "R"

def abbey(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    if len(opponent_history) < 3:
        return "P"

    last = opponent_history[-3:]
    if last == ["R", "R", "R"]:
        return "P"
    elif last == ["P", "P", "P"]:
        return "S"
    elif last == ["S", "S", "S"]:
        return "R"
    return random.choice(["R", "P", "S"])

def kris(prev_play, _=[]):
    return random.choice(["R", "P", "S"])

def mrugesh(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    if len(opponent_history) < 3:
        guess = "R"
    else:
        last_three = "".join(opponent_history[-3:])
        guess = max(set(last_three), key=last_three.count)

    counter_moves = {"R": "P", "P": "S", "S": "R"}
    return counter_moves[guess]

### Game Engine ###
def play(player1, player2, num_games=1000, verbose=False):
    p1_prev = ""
    p2_prev = ""

    p1_score = 0
    p2_score = 0

    for i in range(num_games):
        move1 = player1(p2_prev)
        move2 = player2(p1_prev)

        if move1 == move2:
            result = "Tie"
        elif (move1 == "R" and move2 == "S") or \
             (move1 == "P" and move2 == "R") or \
             (move1 == "S" and move2 == "P"):
            p1_score += 1
            result = "Player 1 wins"
        else:
            p2_score += 1
            result = "Player 2 wins"

        p1_prev = move1
        p2_prev = move2

        if verbose:
            print(f"Game {i+1}: P1: {move1} | P2: {move2} => {result}")

    print(f"\nFinal Score after {num_games} games:")
    print(f"Player 1 (You): {p1_score}")
    print(f"Player 2 (Opponent): {p2_score}")
    if p1_score > p2_score:
        print("You Win! 🎉")
    elif p2_score > p1_score:
        print("You Lose! 😢")
    else:
        print("It's a Tie!")

### Main Test ###
if __name__ == "__main__":
    print("Testing against Quincy:")
    play(player, quincy, 1000)

    print("\nTesting against Abbey:")
    play(player, abbey, 1000)

    print("\nTesting against Kris:")
    play(player, kris, 1000)

    print("\nTesting against Mrugesh:")
    play(player, mrugesh, 1000)
