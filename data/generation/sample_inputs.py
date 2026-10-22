import json
import random
import sys
from pathlib import Path

random.seed(42)

CATEGORIES = {
    "protein": [
        "eggs", "tofu", "leftover roast chicken", "canned tuna", "sausages", "bacon",
        "ham", "salmon fillet", "shrimp", "ground beef", "pork chops", "ground lamb",
        "chickpeas", "black beans", "red lentils", "hummus",
    ],
    "starch": [
        "cooked rice", "pasta", "potatoes", "sweet potato", "stale bread", "tortillas",
        "rice noodles", "quinoa", "rolled oats", "puff pastry", "flour", "corn",
    ],
    "vegetable": [
        "spinach", "carrots", "zucchini", "kimchi", "tomatoes", "onions", "mushrooms",
        "bell pepper", "eggplant", "cabbage", "cucumber", "frozen peas", "broccoli",
        "cauliflower", "leek", "beetroot", "pumpkin", "avocado", "canned tomatoes",
        "olives", "pickles",
    ],
    "dairy": [
        "feta", "mozzarella", "plain yogurt", "cheddar", "parmesan", "heavy cream",
        "butter", "milk", "cream cheese", "goat cheese", "coconut milk",
    ],
    "flavor": [
        "garlic", "fresh ginger", "lemon", "soy sauce", "miso paste", "harissa",
        "mustard", "tahini", "peanut butter", "honey", "fresh basil", "fresh cilantro",
        "mint", "chili flakes", "capers", "smoked paprika", "cumin", "curry powder",
    ],
    "wildcard": [
        "apples", "bananas", "pineapple", "mango", "oranges", "peanuts", "walnuts",
        "dates", "raisins",
    ],
}

CUISINES = [
    "Korean", "Ethiopian", "Peruvian", "Lebanese", "Vietnamese", "Nigerian",
    "Georgian", "Mexican", "Japanese", "Indian", "Thai", "Moroccan", "Turkish",
    "Greek", "Italian", "French", "Spanish", "Brazilian", "Indonesian", "Filipino",
    "Iranian", "Senegalese", "Ghanaian", "Sichuan Chinese", "Argentinian",
    "Polish", "Swedish", "Jamaican",
]

TWISTS = [
    "fusion of two cuisines",
    "a technique from one cuisine applied to an ingredient from another",
    "an unexpected reinvention of a well-known dish",
    "a surprising contrast of textures",
    "a street-food style reinvention",
]

APPLIANCE_SETS = [
    ["pan"], ["pot"], ["microwave"], ["oven"],
    ["pan", "microwave"], ["pot", "pan"], ["pot", "microwave"],
    ["oven", "pan"], ["blender", "pan"], ["air fryer", "pan"],
    ["rice cooker", "pan"], ["oven", "pot", "pan"],
]

TIMES = [15, 20, 30, 45, 60]


def sample_fridge():
    items = [random.choice(CATEGORIES["protein"]), random.choice(CATEGORIES["starch"])]
    items += random.sample(CATEGORIES["vegetable"], random.randint(2, 3))
    if random.random() < 0.6:
        items.append(random.choice(CATEGORIES["dairy"]))
    items += random.sample(CATEGORIES["flavor"], random.randint(1, 2))
    if random.random() < 0.2:
        items.append(random.choice(CATEGORIES["wildcard"]))
    items = list(dict.fromkeys(items))  # remove duplicates, keep order
    random.shuffle(items)
    return items


def sample_input(i):
    cuisine = "surprise" if random.random() < 0.1 else random.choice(CUISINES)
    return {
        "id": f"r_{i:05d}",
        "ingredients": sample_fridge(),
        "max_time_min": random.choice(TIMES),
        "appliances": random.choice(APPLIANCE_SETS),
        "cuisine": cuisine,
        "twist": random.choice(TWISTS),
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out = Path("data/raw/inputs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i in range(n):
            f.write(json.dumps(sample_input(i)) + "\n")
    print(f"{n} inputs written to {out}")
