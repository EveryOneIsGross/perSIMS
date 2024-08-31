import random
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
from openai import OpenAI
import json
import os
from typing import Union
import time
import yaml
import argparse
import csv

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

class Mood(str, Enum):
    HAPPY = "happy"
    SATISFIED = "satisfied"
    STRESSED = "stressed"

class Item(BaseModel):
    symbol: str
    name: str
    stats: Dict[str, float]
    price: int
    operating_cost: int = 0
    energy_cost: float = 0
    weight: float = 1
    description: str = ""
    pay_rate: int = 0

class Sim(BaseModel):
    x: int = Field(0, ge=0)
    y: int = Field(0, ge=0)
    needs: Dict[str, float] = Field(
        default_factory=lambda: {
            "hunger": 10, "hygiene": 10, "bladder": 10, "energy": 10,
            "social": 10, "fun": 10, "environment": 10, "comfort": 10
        }
    )
    mood: Mood = Mood.HAPPY
    money: float = Field(100, ge=0)
    inventory: Dict[str, int] = Field(default_factory=dict)
    current_item: Optional[Item] = None
    item_interaction_turns: int = Field(0, ge=0)
    sleeping: bool = False
    sleep_turns: int = Field(0, ge=0)
    days: int = Field(0, ge=0)
    journal: Dict[int, List[Dict[str, str]]] = Field(default_factory=dict)

    def update_needs(self, item: Item):
        for stat, value in item.stats.items():
            self.needs[stat] = min(10, self.needs[stat] + value * 0.5)
        
        if item.energy_cost:
            self.needs["energy"] = max(0, self.needs["energy"] - item.energy_cost * 0.1)

    def update_mood(self):
        total_needs = sum(self.needs.values())
        if total_needs < 50:
            self.mood = Mood.STRESSED
        elif total_needs < 70:
            self.mood = Mood.SATISFIED
        else:
            self.mood = Mood.HAPPY

    def add_journal_entry(self, activity: str):
        if self.days not in self.journal:
            self.journal[self.days] = []
        self.journal[self.days].append({"activity": activity, "mood": self.mood})

class House(BaseModel):
    width: int
    height: int
    map: List[List[Dict[str, str]]] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self.initialize_map()

    def initialize_map(self):
        self.map = [[{"symbol": "·", "name": "Floor"} for _ in range(self.width)] for _ in range(self.height)]
        for x in range(self.width):
            self.map[0][x] = self.map[self.height-1][x] = {"symbol": "█", "name": "Wall"}
        for y in range(self.height):
            self.map[y][0] = self.map[y][self.width-1] = {"symbol": "█", "name": "Wall"}

    def place_item(self, item: Item):
        while True:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.map[y][x]["symbol"] == "·":
                self.map[y][x] = {"symbol": item.symbol, "name": item.name, "item": item}
                break

class Market(BaseModel):
    items: Dict[str, Dict[str, Union[str, int, float]]] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.items:
            self.initialize_items()

    def initialize_items(self):
        self.items = {
            "food": {"name": "Food", "price": 10, "quantity": 100, "related_stat": "hunger"},
            "electronics": {"name": "Electronics", "price": 50, "quantity": 50, "related_stat": "fun"},
            "furniture": {"name": "Furniture", "price": 100, "quantity": 25, "related_stat": "comfort"}
        }

class Simulation(BaseModel):
    sim: Sim = Field(default_factory=Sim)
    house: House = Field(default_factory=lambda: House(width=16, height=8))
    market: Market = Field(default_factory=Market)
    items: Dict[str, Item] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    current_activity: str = "idle"
    current_situation: str = Field(default="You are neither here nor there.")
    activity_duration: int = 0
    log_file_path: str = "simulation_log.jsonl"
    client: Optional[OpenAI] = None  # Add this line
    config_file: str = "config.yaml"
    model: str = "hermes3"
    system_prompt: str = Field(default="""
You are a Sim in an economic simulation. Your responses should reflect the persona of a Sim living in a humble home, dealing with various needs and economic constraints.

Your current state is:
{{SIM_STATE}}

When responding, use the following format:
<simthinking>
• Current state: [Describe your current state]
• Needs assessment: [Evaluate your current needs]
• Emotional check: [Reflect on your current mood]
• Short-term plan: [What do you plan to do next?]
• Long-term considerations: [Any thoughts about your overall situation?]
</simthinking>
                               
Check whether you are 'moving to' or 'using' an item in your house, and describe your current actions and thoughts only.
After your self-reflection, provide a brief statement or action as the Sim would express it. 
""")
    user_prompt: str = Field(default="""
Current situation: {current_item_desc}
Current activity: {current_activity}

Please provide a self-reflection on your current state, needs, and plans.
""")
    metrics_log_file: str = "simulation_metrics.csv"
    step_count: int = 0  # Add this line to track steps

    class Config:
        arbitrary_types_allowed = True

    @validator('*', pre=True, always=True)
    def add_methods(cls, v):
        return v

    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
            self.system_prompt = config.get('system_prompt', self.system_prompt)
            self.user_prompt = config.get('user_prompt', self.user_prompt)
            print(f"Loaded user_prompt: {self.user_prompt}")  # Debug print
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found. Using default prompts.")
        except yaml.YAMLError:
            print(f"Error parsing {self.config_file}. Using default prompts.")

    def get_llm_response(self, user_input: str) -> str:
        # Replace the {{SIM_STATE}} placeholder with the actual sim state
        system_prompt = self.system_prompt.replace("{{SIM_STATE}}", self.get_llm_friendly_sim_state_md())

        # Populate the placeholders in the user_prompt
        current_item_desc = self.current_situation if hasattr(self, 'current_situation') else (
            self.sim.current_item.description if self.sim.current_item 
            else "You are not interacting with any item at the moment."
        )
        populated_user_prompt = self.user_prompt.replace("{{current_item_desc}}", current_item_desc)
        populated_user_prompt = populated_user_prompt.replace("{{current_activity}}", self.current_activity)

        # Add conversation history to the prompt
        conversation_context = "\n\nPrevious conversation:\n"
        for entry in self.conversation_history[-5:]:  # Include last 5 entries
            role = "Sim" if entry["role"] == "assistant" else "Human"

            content = entry["content"]
            conversation_context += f"{role}: {content}\n"

        # Combine all parts of the prompt
        full_prompt = f"{system_prompt}\n\n{conversation_context}\n\n{populated_user_prompt}"

        print(Colors.CYAN + "Full Prompt:" + Colors.RESET)
        print(full_prompt)

        messages = [
            {"role": "user", "content": full_prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        assistant_response = response.choices[0].message.content
        self.log_interaction_to_jsonl(system_prompt, populated_user_prompt, assistant_response)

        # Print the LLM response in magenta
        print(Colors.MAGENTA + "LLM Response:" + Colors.RESET)
        print(Colors.MAGENTA + assistant_response + Colors.RESET)

        return assistant_response

    def __init__(self, **data):
        super().__init__(**data)
        self.load_config()
        self.initialize_items()
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.load_state()
        self.generate_house()

    def load_state(self):
        if os.path.exists(self.log_file_path):
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if 'sim_state' in entry:
                        self.sim = Sim.parse_obj(entry['sim_state'])
                        self.conversation_history = entry.get('conversation_history', [])
                        self.step_count = entry.get('step', 0)
                        break  # We only need the last state
        else:
            self.sim = Sim()
            self.conversation_history = []
            self.step_count = 0


    # Script 1
    def initialize_items(self):
        self.items = {
            'F': Item(symbol='F', name='Fridge', stats={"hunger": 8}, price=500, operating_cost=5, energy_cost=1, weight=2, 
                    description="You are in the small kitchen, standing in front of the fridge. The cool air rushes out as you open it, promising to satisfy your hunger."),
            'S': Item(symbol='S', name='Shower', stats={"hygiene": 8, "comfort": 2, "environment": 1}, price=300, energy_cost=1, weight=2, 
                    description="You are in the compact bathroom, preparing to step into the shower. The thought of warm water cascading over you is already refreshing."),
            'T': Item(symbol='T', name='Toilet', stats={"bladder": 10}, price=200, energy_cost=1, weight=3, 
                    description="You are in the modest bathroom, approaching the toilet. It's not glamorous, but it's a necessary part of life in this humble home."),
            'B': Item(symbol='B', name='Bed', stats={"environment": 2, "energy": 8, "comfort": 6}, price=800, weight=3, 
                    description="You are in the cozy bedroom, looking at your inviting bed. The soft pillows and warm blanket promise a good night's sleep."),
            'V': Item(symbol='V', name='TV', stats={"fun": 10, "comfort": 2, "environment": 1}, price=400, operating_cost=3, energy_cost=1, weight=1, 
                    description="You are in the simple living room, settling down in front of the TV. The flickering screen offers a window to entertainment and relaxation."),
            'P': Item(symbol='P', name='Painting', stats={"environment": 10, "fun": 5}, price=300, energy_cost=1, weight=1, 
                    description="You are in the modest living room, admiring the painting on the wall. Though small, it adds a touch of color and personality to your home."),
            'C': Item(symbol='C', name='Couch', stats={"environment": 1, "comfort": 8, "fun": 1, "energy": 1}, price=600, weight=1, 
                    description="You are in the comfortable living room, sinking into the worn but cozy couch. It's your favorite spot to unwind after a long day."),
            'H': Item(symbol='H', name='Phone', stats={"social": 4, "fun": 4}, price=200, energy_cost=1, weight=1, 
                    description="You are in your quiet bedroom, reaching for your phone on the nightstand. It's your connection to the outside world from this small, humble home."),
            'W': Item(symbol='W', name='Work', stats={"social": 5, "environment": 5, "comfort": 10, "fun": 10}, 
                      price=0, energy_cost=0, weight=64, 
                      description="You are in your makeshift home office corner, preparing to start work. The simple desk and chair aren't much, but they're all you need to earn a living.",
                      pay_rate=30)
        }

    def generate_house(self):
        for item in self.items.values():
            self.house.place_item(item)
        self.place_sim()

    def place_sim(self):
        while True:
            self.sim.x = random.randint(1, self.house.width - 2)
            self.sim.y = random.randint(1, self.house.height - 2)
            if self.house.map[self.sim.y][self.sim.x]["symbol"] == "·":
                break

    def interact_with_item(self, item: Item):
        self.sim.current_item = item
        self.sim.item_interaction_turns = 0
        self.sim.update_needs(item)
        
        if item.operating_cost:
            self.sim.money -= item.operating_cost
            print(f"The Sim paid ${item.operating_cost} to use the {item.name}.")
        
        self.current_activity = f"using {item.name}"
        self.activity_duration = 0
        print(f"The Sim starts using the {item.name}.")
        
        if item.name == 'Fridge':
            self.update_market_volume('F')
        
        if item.name == 'Work':
            self.work()
        
        self.check_debt()


    def continue_item_interaction(self):
        if not self.sim.current_item:
            return

        self.sim.item_interaction_turns += 1
        self.activity_duration += 1
        
        self.current_activity = f"using {self.sim.current_item.name}"
        self.current_situation = self.sim.current_item.description
        
        # Update needs based on the item's stats
        for stat, value in self.sim.current_item.stats.items():
            self.sim.needs[stat] = min(10, self.sim.needs[stat] + value * 0.5)
        
        # Decrease all needs except those addressed by the current item
        for need in self.sim.needs:
            if need not in self.sim.current_item.stats:
                self.sim.needs[need] = max(0, self.sim.needs[need] - 0.5)

        # Apply energy cost if applicable
        if self.sim.current_item.energy_cost:
            self.sim.needs["energy"] = max(0, self.sim.needs["energy"] - self.sim.current_item.energy_cost * 0.1)

        print(f"The Sim continues using the {self.sim.current_item.name}. (Turn {self.sim.item_interaction_turns})")

        if self.sim.current_item.name == 'Work':
            earned_money = self.sim.current_item.pay_rate
            self.sim.money += earned_money
            print(f"The Sim earned ${earned_money:.2f} from work.")
            
            produced_item = self.produce_market_item()
            print(f"The Sim produced {produced_item['quantity']} {produced_item['name']} for the market.")

            # End work session if energy is too low or after a certain number of turns
            if self.sim.needs["energy"] <= 2 or self.sim.item_interaction_turns >= 5:
                print("The Sim is too tired to continue working.")
                self.sim.current_item = None
                self.sim.item_interaction_turns = 0
                return

        # Check if the primary need is fully satisfied or if the interaction has lasted too long
        primary_need = next(iter(self.sim.current_item.stats))
        if self.sim.needs[primary_need] >= 9.5 or self.sim.item_interaction_turns >= 10:
            print(f"The Sim finishes using the {self.sim.current_item.name}.")
            self.sim.add_journal_entry(f"finished using {self.sim.current_item.name}")
            
            self.sim.current_item = None
            self.sim.item_interaction_turns = 0

    def work(self):
        print("The Sim starts working.")
        self.sim.current_item = self.items['W']
        self.sim.item_interaction_turns = 0

    def adjust_pay_rate(self, adjustment: float):
        work_item = self.items['W']
        work_item.pay_rate = max(0, work_item.pay_rate + adjustment)
        print(f"The Sim's pay rate has been adjusted to ${work_item.pay_rate:.2f}/hr.")

    def produce_market_item(self):
        items = list(self.market.items.keys())
        produced_item = random.choice(items)
        quantity = random.randint(1, 5)
        
        self.market.items[produced_item]["quantity"] += quantity
        
        return {
            "name": self.market.items[produced_item]["name"],
            "quantity": quantity
        }

    def update_market_volume(self, item_key: str):
        market_category = {
            'F': 'food', 'V': 'electronics', 'H': 'electronics',
            'C': 'furniture', 'B': 'furniture', 'P': 'furniture'
        }.get(item_key)
        
        if market_category:
            self.market.items[market_category]["quantity"] = max(0, self.market.items[market_category]["quantity"] - 1)
            self.market.items[market_category]["price"] = min(1000, int(self.market.items[market_category]["price"] * 1.05))

    def check_debt(self):
        if self.sim.money < 0:
            print("The Sim has gone into debt!")
            while self.sim.money < 0:
                item_to_sell = self.find_item_to_sell()
                if item_to_sell:
                    self.sell_item(item_to_sell)
                else:
                    print("The Sim has no more items to sell and remains in debt!")
                    break

    def find_item_to_sell(self) -> Optional[Tuple[int, int, Item]]:
        sellable_symbols = {'V', 'P', 'C'}  # TV, Painting, Couch
        item_counts = {}
        sellable_items = []

        # First pass: count items and identify sellable items
        for y in range(1, self.house.height - 1):
            for x in range(1, self.house.width - 1):
                item = self.house.map[y][x].get("item")
                if item:
                    item_counts[item.symbol] = item_counts.get(item.symbol, 0) + 1
                    if item.symbol in sellable_symbols or item_counts[item.symbol] > 1:
                        sellable_items.append((x, y, item))

        # Second pass: prioritize selling duplicates
        for x, y, item in sellable_items:
            if item_counts[item.symbol] > 1:
                return x, y, item

        # If no duplicates, sell any remaining sellable item
        return sellable_items[0] if sellable_items else None

    def sell_item(self, item_location: Tuple[int, int, Item]):
        x, y, item = item_location
        sell_price = item.price // 2
        self.sim.money += sell_price
        self.house.map[y][x] = {"symbol": "·", "name": "Floor"}
        print(f"The Sim sold the {item.name} for ${sell_price} to cover debt.")
        
        # Update item counts
        item_symbol = item.symbol
        item_counts = sum(1 for cell in sum(self.house.map, []) if cell.get("item") and cell["item"].symbol == item_symbol)
        if item_counts == 0:
            print(f"Warning: The Sim has sold their only {item.name}!")

    def check_for_automatic_purchase(self):
            lowest_stat = min(self.sim.needs, key=self.sim.needs.get)
            if self.sim.needs[lowest_stat] < 3 and self.sim.money >= 1000:
                item_to_buy = {
                    'hunger': 'F', 'fun': 'V', 'comfort': 'C', 'environment': 'P'
                }.get(lowest_stat)
                
                if item_to_buy and not self.find_item_for_need(lowest_stat):
                    self.buy_item(item_to_buy)
                    print(f"The Sim automatically bought a {self.items[item_to_buy].name} to address low {lowest_stat}.")



    def buy_item(self, item_key: str):
        item = self.items[item_key]
        if self.sim.money >= item.price:
            self.sim.money -= item.price
            self.house.place_item(item)
            print(f"The Sim bought a {item.name} for ${item.price}.")
            self.update_market_volume(item_key)
        else:
            print(f"The Sim can't afford the {item.name}.")

    def find_item_for_need(self, need: str) -> Optional[Tuple[int, int]]:
        for y in range(1, self.house.height - 1):
            for x in range(1, self.house.width - 1):
                item = self.house.map[y][x].get("item")
                if item and (need in item.stats or (need == "work" and item.name == "Work")):
                    return x, y
        return None

    def choose_item(self):
        sorted_needs = sorted(
            [(need, value, self.get_item_weight(need)) for need, value in self.sim.needs.items()],
            key=lambda x: (x[1], -x[2])  # Sort by need value (ascending) and item weight (descending)
        )
        return sorted_needs[0][0]  # Return the need with lowest value and highest item weight

    def get_item_weight(self, need: str) -> float:
        items = [item for item in self.items.values() if need in item.stats]
        return max([item.weight for item in items], default=0)

    def simulate_step(self):
        self.step_count += 1
        daily_expenses = 1
        self.sim.money -= daily_expenses
        print(f"Daily expenses: ${daily_expenses}")

        # Check for automatic purchase at the start of each turn
        self.check_for_automatic_purchase()

        # Update mood and needs
        self.sim.update_mood()
        for need in self.sim.needs:
            self.sim.needs[need] = max(0, self.sim.needs[need] - 0.1)

        # If the sim is currently interacting with an item, continue the interaction
        if self.sim.current_item:
            self.continue_item_interaction()
        else:
            # Choose the most urgent need and find the corresponding item
            target_need = self.choose_item()
            target_item = self.find_item_for_need(target_need)

            if target_item:
                # Move towards the target item
                dx = 1 if target_item[0] > self.sim.x else -1 if target_item[0] < self.sim.x else 0
                dy = 1 if target_item[1] > self.sim.y else -1 if target_item[1] < self.sim.y else 0
                self.move_sim(dx, dy)
                
                self.current_activity = f"moving towards {self.house.map[target_item[1]][target_item[0]]['name']}"
                self.current_situation = f"You are heading towards {self.house.map[target_item[1]][target_item[0]]['name']} to fulfill your {target_need} need."
                
                # Check if we've reached the target
                if (self.sim.x, self.sim.y) == target_item:
                    item = self.house.map[self.sim.y][self.sim.x].get("item")
                    if item:
                        self.interact_with_item(item)
            else:
                # If no target item is found, wander randomly
                self.move_sim(random.randint(-1, 1), random.randint(-1, 1))
                self.current_activity = "wandering"
                self.current_situation = "You are neither here nor there."

        self.activity_duration += 1
        self.log_metrics()

    def check_for_automatic_purchase(self):
        PURCHASE_THRESHOLD = 1000  # Set this to your desired value
        sorted_needs = sorted(self.sim.needs.items(), key=lambda x: x[1])
        lowest_need, lowest_value = sorted_needs[0]

        if self.sim.money >= PURCHASE_THRESHOLD:
            item_to_buy = None
            for item_key, item in self.items.items():
                if lowest_need in item.stats and item.price <= self.sim.money:
                    item_to_buy = item_key
                    break

            if item_to_buy:
                self.buy_item(item_to_buy)
                print(f"The Sim automatically bought a {self.items[item_to_buy].name} to address low {lowest_need}.")
        
    def move_sim(self, dx: int, dy: int):
        new_x = self.sim.x + dx
        new_y = self.sim.y + dy
        if 0 < new_x < self.house.width - 1 and 0 < new_y < self.house.height - 1:
            self.sim.x = new_x
            self.sim.y = new_y
            for need in self.sim.needs:
                self.sim.needs[need] = max(0, self.sim.needs[need] - 0.1)
            
            item = self.house.map[self.sim.y][self.sim.x].get("item")
            if item:
                self.interact_with_item(item)
            else:
                self.current_activity = "wandering"
                self.current_situation = "You are neither here nor there."

    def move_towards(self, target: Tuple[int, int]):
        dx = 1 if target[0] > self.sim.x else -1 if target[0] < self.sim.x else 0
        dy = 1 if target[1] > self.sim.y else -1 if target[1] < self.sim.y else 0
        self.move_sim(dx, dy)

    def sleep_where_you_are(self, force_sleep=False):
        nearest_bed = self.find_nearest_bed()

        if self.house.map[self.sim.y][self.sim.x].get("item") and self.house.map[self.sim.y][self.sim.x]["item"].name == 'Bed':
            if not self.sim.sleeping:
                self.sim.sleeping = True
                self.sim.current_item = self.items['B']
                self.sim.item_interaction_turns = 0
                self.sim.days += 1
                print("The Sim has gone to bed. A new day has started.")
        elif nearest_bed and (force_sleep or self.sim.needs["energy"] <= 2 or random.random() < 0.7):
            self.move_towards(nearest_bed)
            print("The Sim is tired and is moving towards the bed.")
        else:
            sleep_location = "on the floor" if self.house.map[self.sim.y][self.sim.x]["symbol"] == "·" else f"on the {self.house.map[self.sim.y][self.sim.x]['name']}"
            print(f"The Sim fell asleep {sleep_location}. It wasn't very comfortable. They'll wake up soon.")
            for need in self.sim.needs:
                if need != "energy":
                    self.sim.needs[need] = max(0, self.sim.needs[need] - 1)
            self.sim.needs["energy"] = min(self.sim.needs["energy"] + 3, 10)
            self.sim.days += 1
            self.add_journal_entry(f"Slept {sleep_location}")

    def find_nearest_bed(self):
        nearest_bed = None
        shortest_distance = float('inf')

        for y in range(1, self.house.height - 1):
            for x in range(1, self.house.width - 1):
                if self.house.map[y][x].get("item") and self.house.map[y][x]["item"].name == 'Bed':
                    distance = abs(self.sim.x - x) + abs(self.sim.y - y)
                    if distance < shortest_distance:
                        shortest_distance = distance
                        nearest_bed = (x, y)

        return nearest_bed

    def get_ansi_house_layout(self) -> str:
        color_map = {
            "█": Colors.WHITE,  # Wall
            "·": Colors.RESET,  # Floor
            "F": Colors.CYAN,   # Fridge
            "S": Colors.BLUE,   # Shower
            "T": Colors.YELLOW, # Toilet
            "B": Colors.MAGENTA,# Bed
            "V": Colors.GREEN,  # TV
            "P": Colors.RED,    # Painting
            "C": Colors.GREEN,  # Couch
            "H": Colors.BLUE,   # Phone
            "W": Colors.YELLOW  # Work
        }
        
        layout = ""
        for y, row in enumerate(self.house.map):
            for x, cell in enumerate(row):
                if x == self.sim.x and y == self.sim.y:
                    layout += f"{Colors.RED}@{Colors.RESET}"
                else:
                    symbol = cell["symbol"]
                    layout += f"{color_map.get(symbol, Colors.RESET)}{symbol}{Colors.RESET}"
            layout += "\n"
        return layout

    def get_ansi_bar_graph(self, value: float, max_value: float = 10, width: int = 20) -> str:
        filled_width = int(value / max_value * width)
        bar = "█" * filled_width + "░" * (width - filled_width)
        color = Colors.GREEN if value > max_value * 0.6 else Colors.YELLOW if value > max_value * 0.3 else Colors.RED
        return f"{color}{bar}{Colors.RESET}"

    def get_ansi_stats(self) -> str:
        stats = f"{Colors.CYAN}Sim Stats:{Colors.RESET}\n"
        for need, value in self.sim.needs.items():
            stats += f"{need.capitalize():>12}: {self.get_ansi_bar_graph(value)} {value:.1f}\n"
        stats += f"{Colors.YELLOW}{'Money':>12}: ${self.sim.money}{Colors.RESET}\n"
        stats += f"{Colors.MAGENTA}{'Mood':>12}: {self.sim.mood}{Colors.RESET}\n"
        stats += f"{Colors.BLUE}{'Activity':>12}: {self.current_activity}{Colors.RESET}\n"
        stats += f"{Colors.GREEN}{'Duration':>12}: {self.activity_duration} steps{Colors.RESET}\n"
        return stats

    def get_ansi_display(self) -> str:
        house_layout = self.get_ansi_house_layout()
        stats = self.get_ansi_stats()
        
        # Combine stats and house layout side by side
        stats_lines = stats.split("\n")
        house_lines = house_layout.split("\n")
        max_lines = max(len(stats_lines), len(house_lines))
        max_stats_width = max(len(line) for line in stats_lines)
        
        display = ""
        for i in range(max_lines):
            stats_line = stats_lines[i] if i < len(stats_lines) else ""
            house_line = house_lines[i] if i < len(house_lines) else ""
            display += f"{stats_line:<{max_stats_width}}    {house_line}\n"
        
        return display

    def get_sim_state_md(self) -> str:
        return f"""
# Sim State
{self.get_ansi_display()}

## Market State:
{json.dumps(self.market.items, indent=2)}
"""

    def load_chat_history(self):
        history = []
        if os.path.exists(self.log_file_path):
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    history.append({"role": "user", "content": entry["user_input"]})
                    history.append({"role": "assistant", "content": entry["assistant_response"]})
        return history[-20:]  # Keep only the last 20 messages to limit context size

    def log_interaction_to_jsonl(self, system_prompt, user_input, assistant_response):
        log_entry = {
            "step": self.step_count,
            "system_prompt": system_prompt,
            "user_input": user_input,
            "assistant_response": assistant_response
        }
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")

    def get_llm_friendly_sim_state_md(self) -> str:
        # Format needs
        needs_md = "## Sim Needs\n"
        for need, value in self.sim.needs.items():
            status = "DESPERATE" if value == 0 else "SATIATED" if value == 10 else ""
            needs_md += f"- {need.capitalize()}: {status} {value:.1f}/10\n"

        # Format house layout
        house_layout = []
        for row in self.house.map:
            layout_row = []
            for cell in row:
                if cell['symbol'] == '·':
                    layout_row.append('.')  # Empty space
                elif cell['symbol'] == '█':
                    layout_row.append('#')  # Wall
                else:
                    layout_row.append(cell['symbol'])  # Item
            house_layout.append(layout_row)

        # Mark Sim's position
        house_layout[self.sim.y][self.sim.x] = '@'

        house_md = "## House Layout\n```\n"
        for row in house_layout:
            house_md += ''.join(row) + "\n"
        house_md += "```\n"
        key_md = "## House Layout Key\n"
        key_md += "- `.`: Empty space\n"
        key_md += "- `#`: Wall\n"
        key_md += "- `@`: Sim's position\n"
        for item in self.items.values():
            key_md += f"- `{item.symbol}`: {item.name}\n"
        house_md += key_md

        # Format other stats
        stats_md = f"""## Sim Stats
- Money: ${self.sim.money}
- Mood: {self.sim.mood}
- Current Activity: {self.current_activity}
- Activity Duration: {self.activity_duration} steps
- Days: {self.sim.days}

"""

        # Add current item description if applicable
        if self.sim.current_item:
            stats_md += f"\n## Current Interaction\n{self.sim.current_item.description}\n"

        # Format market state
        market_md = "## Market State\n"
        for item, details in self.market.items.items():
            market_md += f"- {item.capitalize()}: Price ${details['price']}, Quantity {details['quantity']}\n"

        # Combine all sections
        return f"""# Sim State

{needs_md}
{stats_md}
{house_md}
{market_md}
"""

    def run_simulation_with_llm(self, num_responses=None):
        print("Welcome to the Economic Simulation. The simulation will run automatically.")
        
        response_count = 0
        while num_responses is None or response_count < num_responses:
            self.simulate_step()
            print(self.get_ansi_display())
            
            llm_response = self.get_llm_response(self.user_prompt)
            #print("Sim's thoughts:", llm_response)

            self.update_conversation_history("assistant", llm_response)
            self.save_state()  # Save state after each interaction

            response_count += 1
            
            # Add a short delay between steps to make the output readable
            time.sleep(2)

        print("Simulation complete.")

    def save_state(self):
        state = {
            'sim_state': self.sim.dict(),
            'conversation_history': self.conversation_history,
            'step': self.step_count
        }
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
            f.write("\n")

    def update_conversation_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.conversation_history = self.conversation_history[-20:]  # Keep only the last 20 messages

    def run_simulation_without_llm(self, num_steps=None):
        print("Welcome to the Economic Simulation. Running in zero interaction mode.")
        
        step_count = 0
        while num_steps is None or step_count < num_steps:
            self.simulate_step()
            print(self.get_ansi_display())
            
            print(f"Step {step_count + 1} completed.")
            self.save_state()  # Save state after each step

            step_count += 1
            
            # Add a short delay between steps to make the output readable
            time.sleep(0.1)

        print("Simulation complete.")

    def log_metrics(self):
        metrics = {
            "step": self.step_count,
            "day": self.sim.days,
            "x": self.sim.x,
            "y": self.sim.y,
            "money": self.sim.money,
            "mood": self.sim.mood.value,
            "current_activity": self.current_activity,
            "activity_duration": self.activity_duration,
        }

        # Add needs
        for need, value in self.sim.needs.items():
            metrics[f"need_{need}"] = value

        # Add market data
        for item, details in self.market.items.items():
            metrics[f"market_{item}_price"] = details['price']
            metrics[f"market_{item}_quantity"] = details['quantity']

        # Write to CSV
        file_exists = os.path.isfile(self.metrics_log_file)
        with open(self.metrics_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

def main():
    parser = argparse.ArgumentParser(description="Run the Economic Simulation")
    parser.add_argument("--model", default="hermes3", help="LLM model to use")
    parser.add_argument("--responses", type=int, help="Number of responses to generate (default: continuous)")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--zeroint", action="store_true", help="Run in zero interaction mode (disable LLM calls)")
    parser.add_argument("--turns", type=int, help="Number of turns to run for")
    
    args = parser.parse_args()

    sim = Simulation(config_file=args.config)
    sim.model = args.model

    if args.zeroint:
        sim.run_simulation_without_llm(num_steps=args.turns or args.responses)
    else:
        sim.run_simulation_with_llm(num_responses=args.turns or args.responses)

if __name__ == "__main__":
    main()
