from flask import Flask, render_template, request, jsonify
import random
import requests
import json
import threading
import time
import queue
from collections import deque
import tensorflow as tf
import numpy as np
import math
# Load the direction model
direction_model = None

# Add a global variable to store the AI mode
ai_mode = 'llm'  # Default to LLM

app = Flask(__name__)

# Game state
game_state = {
    'board_size': {'width': 30, 'height': 20},
    'human': {'x': 0, 'y': 10, 'position': 'K1'},
    'ai_bots': [],
    'projectiles': [],
    'goal': {'x': 29, 'y': 10},
    'game_over': False,
    'winner': None,
    'tile_labels': {}  # Initialized below
}

# AI action histories
ai_histories = {}

# Lock for thread-safe access to game state
state_lock = threading.Lock()

# Queue for AI actions
ai_action_queue = queue.Queue()

# Define valid actions
VALID_ACTIONS = [
    'move up',
    'move down',
    'move left',
    'move right',
    'shoot up',
    'shoot down',
    'shoot left',
    'shoot right'
]

def load_direction_model():
    global direction_model
    if direction_model is None:
        try:
            direction_model = tf.keras.models.load_model('direction_model.h5', compile=False)
            direction_model.compile(optimizer='adam', loss='mse')  # Compile with default settings
        except Exception as e:
            print(f"Error loading direction model: {e}")
            return None
    return direction_model



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_game', methods=['POST'])
def start_game():
    num_bots = int(request.json['num_bots'])
    with state_lock:
        game_state['tile_labels'] = generate_tile_labels()
        game_state['ai_bots'] = [
            {
                'id': i,
                'x': random.randint(20, 28),
                'y': random.randint(0, 19),
                'position': get_tile_label(random.randint(20, 28), random.randint(0, 19)),
                'facing': 'left'  # Add this line
            }
            for i in range(num_bots)
        ]
        game_state['human']['x'] = 0
        game_state['human']['y'] = 10
        game_state['human']['position'] = 'K1'
        game_state['game_over'] = False
        game_state['winner'] = None
        game_state['projectiles'] = []
    for bot in game_state['ai_bots']:
        ai_histories[bot['id']] = deque(maxlen=5)
    # Start background threads if not already running
    threading.Thread(target=ai_update_loop, daemon=True).start()
    threading.Thread(target=projectile_update_loop, daemon=True).start()
    threading.Thread(target=process_ai_actions, daemon=True).start()
    return jsonify(game_state)

@app.route('/move_human', methods=['POST'])
def move_human():
    direction = request.json['direction']
    dx, dy = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}.get(direction, (0, 0))
    
    with state_lock:
        new_x = game_state['human']['x'] + dx
        new_y = game_state['human']['y'] + dy
        
        if 0 <= new_x < game_state['board_size']['width'] and 0 <= new_y < game_state['board_size']['height']:
            game_state['human']['x'] = new_x
            game_state['human']['y'] = new_y
            game_state['human']['position'] = get_tile_label(new_x, new_y)
        
        check_game_end()
    
    return jsonify({'success': True})

@app.route('/get_game_state')
def get_game_state():
    with state_lock:
        return jsonify(game_state)

def ai_update_loop():
    while True:
        if game_state['game_over']:
            break
        
        with state_lock:
            for bot in game_state['ai_bots']:
                threading.Thread(target=query_ai_model, args=(bot['id'], game_state.copy()), daemon=True).start()
        
        time.sleep(1)  # AI update interval

def process_ai_actions():
    while True:
        try:
            bot_id, action = ai_action_queue.get(timeout=1)
            with state_lock:
                bot = next((b for b in game_state['ai_bots'] if b['id'] == bot_id), None)
                if bot:
                    process_ai_action(bot, action)
                    ai_histories[bot_id].append(action)
            ai_action_queue.task_done()
        except queue.Empty:
            pass
        if game_state['game_over']:
            break

def projectile_update_loop():
    while True:
        if game_state['game_over']:
            break
        
        with state_lock:
            move_projectiles()
            check_collisions()
            check_game_end()
        
        time.sleep(0.1)  # Projectile update interval

def query_ai_model(bot_id, current_state):
    bot = next(b for b in current_state['ai_bots'] if b['id'] == bot_id)
    human = current_state['human']
    
    bot_x, bot_y = bot['x'], bot['y']
    human_x, human_y = human['x'], human['y']
    
    if ai_mode == 'direction_model':
        model = load_direction_model()
        if model is None:
            print(f"Direction model not available. Falling back to LLM for Bot {bot_id}")
        else:
            input_data = np.array([[bot_x/29, bot_y/19, human_x/29, human_y/19]])  # Normalize input
            predicted_angle = model.predict(input_data)[0][0]
            
            print(f"Bot {bot_id} - Raw predicted angle: {predicted_angle}")
            
            # Determine primary axis based on angle magnitude
            if abs(predicted_angle) < 0.05:  # This threshold might need adjustment
                primary_axis = 'horizontal'
            else:
                primary_axis = 'vertical'
            
            # Determine direction based on bot and human positions
            dx = human_x - bot_x
            dy = human_y - bot_y
            
            if primary_axis == 'horizontal':
                primary_action = 'move left' if dx < 0 else 'move right'
            else:
                primary_action = 'move up' if dy < 0 else 'move down'
            
            # Add some randomness to the movement
            if random.random() < 0.2:  # 20% chance to choose a random direction
                action = random.choice(['move up', 'move down', 'move left', 'move right'])
            else:
                action = primary_action
            
            print(f"Bot {bot_id} - Chosen direction: {action}")
            
            # Check if within shooting range
            shooting_range = 5
            distance = math.sqrt(dx**2 + dy**2)
            
            print(f"Bot {bot_id} - Distance to human: {distance}")
            
            if distance <= shooting_range:
                # If in range, alternate between moving and shooting
                if random.choice([True, False]):
                    action = action.replace('move', 'shoot')
            
            ai_action_queue.put((bot_id, action))
            print(f"Bot {bot_id} chose action: {action}")
            return

    # If not using direction model or if it failed, use LLM (existing code)
    # Calculate distance
    distance_x = abs(human_x - bot_x)
    distance_y = abs(human_y - bot_y)
    
    # Define shooting range (adjust as needed)
    shooting_range = 5

    # Determine relative position
    if human_y < bot_y:
        vertical = "above"
    elif human_y > bot_y:
        vertical = "below"
    else:
        vertical = ""

    if human_x < bot_x:
        horizontal = "left"
    elif human_x > bot_x:
        horizontal = "right"
    else:
        horizontal = ""

    relative_position = f"{vertical} {horizontal}".strip()
    if not relative_position:
        relative_position = "same position"

    # Determine if within shooting range
    within_range = distance_x <= shooting_range and distance_y <= shooting_range

    # Construct prompt
    if within_range:
        prompt = f"the player is {relative_position}"
    else:
        prompt = f"the player is {relative_position} and not within shooting range"

    try:
        payload = {
            "model": "player0:latest",
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 10,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "top_p": 1.0,
            "stop": ["\n", "."],
            "stream": False,
        }
        response = requests.post('http://localhost:11434/api/generate', json=payload)
        response.raise_for_status()
        data = response.json()
        if 'response' in data:
            action = data['response'].strip().lower()
            if action in VALID_ACTIONS:
                ai_action_queue.put((bot_id, action))
                print(f"Bot {bot_id} chose action: {action}")  # Debug print
            else:
                print(f"Bot {bot_id} returned invalid action: '{action}'. No action taken.")  # Debug print
        else:
            print(f"Bot {bot_id} received no response from AI model. No action taken.")  # Debug print
    except Exception as e:
        print(f"Error querying AI model for Bot {bot_id}: {e}")
        # Do nothing in case of error

@app.route('/set_ai_mode', methods=['POST'])
def set_ai_mode():
    global ai_mode
    ai_mode = request.json['mode']
    if ai_mode == 'direction_model':
        if load_direction_model() is None:
            return jsonify({'success': False, 'error': 'Failed to load direction model'})
    return jsonify({'success': True})

def smart_default_action(bot, human):
    dx = human['x'] - bot['x']
    dy = human['y'] - bot['y']
    
    if abs(dx) > abs(dy):
        return 'move right' if dx > 0 else 'move left'
    else:
        return 'move down' if dy > 0 else 'move up'

def format_other_bots(bots, current_bot_id):
    other_bots = [b for b in bots if b['id'] != current_bot_id]
    if not other_bots:
        return "None"
    return ", ".join([f"Bot {b['id']} at {b['position']}" for b in other_bots])

def process_ai_action(bot, action):
    action_parts = action.split()
    if len(action_parts) != 2:
        return

    action_type, direction = action_parts
    if action_type == 'move':
        dx, dy = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}.get(direction, (0, 0))
        new_x = bot['x'] + dx
        new_y = bot['y'] + dy
        if 0 <= new_x < game_state['board_size']['width'] and 0 <= new_y < game_state['board_size']['height']:
            bot['x'] = new_x
            bot['y'] = new_y
            bot['position'] = get_tile_label(new_x, new_y)
            bot['facing'] = direction  # Update facing direction
    elif action_type == 'shoot':
        game_state['projectiles'].append({
            'x': bot['x'],
            'y': bot['y'],
            'direction': direction
        })
        bot['facing'] = direction  # Update facing direction when shooting
    print(f"Bot {bot['id']} {action_type} {direction} from {bot['position']}")  # Debug print

def move_projectiles():
    for projectile in game_state['projectiles']:
        dx, dy = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}.get(projectile['direction'], (0, 0))
        projectile['x'] += dx
        projectile['y'] += dy

def check_collisions():
    for projectile in game_state['projectiles'][:]:
        if (projectile['x'] == game_state['human']['x'] and 
            projectile['y'] == game_state['human']['y']):
            game_state['projectiles'].remove(projectile)
            game_state['game_over'] = True
            game_state['winner'] = 'ai'
            return

    game_state['projectiles'] = [
        p for p in game_state['projectiles']
        if 0 <= p['x'] < game_state['board_size']['width'] and
           0 <= p['y'] < game_state['board_size']['height']
    ]

def check_game_end():
    if game_state['human']['x'] == game_state['goal']['x']:
        game_state['game_over'] = True
        game_state['winner'] = 'human'

def generate_tile_labels():
    labels = {}
    for y in range(game_state['board_size']['height']):
        for x in range(game_state['board_size']['width']):
            label = f"{chr(65 + y)}{x + 1}"
            labels[f"{x},{y}"] = label
    return labels

def get_tile_label(x, y):
    return f"{chr(65 + y)}{x + 1}"

# Initialize tile labels at startup
with state_lock:
    game_state['tile_labels'] = generate_tile_labels()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
