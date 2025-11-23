"""
Simplified synthetic PII data generator for noisy STT transcripts.
Generates realistic training examples with correct span tracking.
"""

import json
import random
from typing import List, Dict, Any, Tuple

# Common names
FIRST_NAMES = [
    "ramesh", "priya", "john", "sarah", "kumar", "ravi", "isha", "amit",
    "nikita", "arjun", "divya", "rohit", "neha", "raj", "pooja", "anil",
    "maya", "david", "emma", "sophia", "james", "michael", "anna", "laura",
    "vikram", "ananya", "sanjay", "meera", "vikas", "sneha"
]

LAST_NAMES = [
    "sharma", "patel", "singh", "kumar", "reddy", "gupta", "brown", "smith",
    "johnson", "williams", "jones", "miller", "davis", "wilson", "moore",
    "taylor", "anderson", "thomas", "jackson", "white", "khanna", "desai",
    "verma", "misra", "chopra", "nair", "iyer"
]

# Cities and locations
CITIES = ["mumbai", "delhi", "bangalore", "hyderabad", "pune", "kolkata",
          "houston", "new york", "san francisco", "seattle", "boston", "london",
          "paris", "dubai", "singapore", "tokyo", "sydney", "toronto", "chicago",
          "chennai", "ahmedabad", "chandigarh", "lucknow", "jaipur"]

LOCATIONS = ["india", "united states", "united kingdom", "france", "germany",
             "japan", "canada", "australia", "brazil", "mexico", "california",
             "texas", "florida", "new york", "alaska", "hawaii", "scotland",
             "ireland", "new zealand", "south africa", "europe", "asia"]

EMAIL_DOMAINS = ["gmail", "yahoo", "hotmail", "outlook", "microsoft", "apple",
                 "google", "facebook", "amazon", "netflix", "uber", "airbnb"]

# Number representations
DIGIT_WORDS = ["oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def digits_to_spoken(digits: str, style: str = "individual") -> str:
    """Convert digits to spoken form."""
    if style == "individual":
        return " ".join(DIGIT_WORDS[int(d)] for d in digits)
    elif style == "paired":
        # Use double forms
        result = []
        for d in digits:
            result.append(DIGIT_WORDS[int(d)])
        return " ".join(result)
    else:
        return " ".join(DIGIT_WORDS[int(d)] for d in digits)


def generate_cc() -> str:
    """Generate credit card in spoken form."""
    cc = "".join(str(random.randint(0, 9)) for _ in range(16))
    style = random.choice(["individual", "paired"])
    return digits_to_spoken(cc, style)


def generate_phone() -> str:
    """Generate phone number in spoken form."""
    phone = "".join(str(random.randint(0, 9)) for _ in range(10))
    return digits_to_spoken(phone)


def generate_email() -> str:
    """Generate email in spoken form."""
    name = random.choice(FIRST_NAMES)
    domain = random.choice(EMAIL_DOMAINS)
    tld = random.choice(["com", "org", "net", "edu", "co", "in"])
    return f"{name} at {domain} dot {tld}"


def generate_person_name() -> str:
    """Generate person name."""
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def generate_date() -> str:
    """Generate date in spoken form."""
    month_names = ["january", "february", "march", "april", "may", "june",
                   "july", "august", "september", "october", "november", "december"]
    day = random.randint(1, 28)
    month = random.choice(month_names)
    year = random.randint(2015, 2024)
    
    # Vary format
    fmt = random.choice([
        f"{month} {day} {year}",
        f"{day} {month} {year}",
        f"{month} the {day} {year}"
    ])
    return fmt


def generate_city() -> str:
    """Generate city name."""
    return random.choice(CITIES)


def generate_location() -> str:
    """Generate location name."""
    return random.choice(LOCATIONS)


def create_utterance() -> Dict[str, Any]:
    """Create a single utterance with randomly placed entities."""
    # Templates with {ENTITY_PLACEHOLDER}
    templates = [
        "contact me at {EMAIL} or {PHONE}",
        "my name is {PERSON_NAME} and i work in {LOCATION}",
        "the credit card number is {CREDIT_CARD}",
        "send payment to {EMAIL} from city {CITY}",
        "call me at {PHONE} or email {EMAIL}",
        "my card {CREDIT_CARD} expires on {DATE}",
        "i live in {CITY} which is in {LOCATION}",
        "{PERSON_NAME} from {LOCATION} can be reached at {PHONE}",
        "the event is on {DATE} in {LOCATION}",
        "charged to card {CREDIT_CARD} from {CITY}",
        "my email is {EMAIL} and phone is {PHONE}",
        "{PERSON_NAME} works in {CITY}",
        "visit us in {LOCATION} or call {PHONE}",
        "meeting date is {DATE} with {PERSON_NAME}",
        "bill to {CREDIT_CARD} at {CITY}",
    ]
    
    # Entity generators
    generators = {
        "EMAIL": generate_email,
        "PHONE": generate_phone,
        "PERSON_NAME": generate_person_name,
        "LOCATION": generate_location,
        "CREDIT_CARD": generate_cc,
        "DATE": generate_date,
        "CITY": generate_city,
    }
    
    # Pick a template and fill in entities
    template = random.choice(templates)
    required_entities = set()
    
    # Find which entities are needed
    for key in generators:
        if "{" + key + "}" in template:
            required_entities.add(key)
    
    # Generate values
    replacements = {}
    for entity_type in required_entities:
        replacements[entity_type] = generators[entity_type]()
    
    # Build text and track spans
    text = template
    entities = []
    
    # Replace placeholders and track positions
    for entity_type in required_entities:
        value = replacements[entity_type]
        placeholder = "{" + entity_type + "}"
        
        # Find position of placeholder in current text
        idx = text.find(placeholder)
        if idx != -1:
            # Calculate actual span positions
            start = idx
            text = text.replace(placeholder, value, 1)  # Replace only first occurrence
            end = start + len(value)
            
            entities.append({
                "start": start,
                "end": end,
                "label": entity_type
            })
    
    # Add noise: remove some punctuation randomly
    if random.random() < 0.3:
        text = text.replace(".", "")
    if random.random() < 0.2:
        text = text.replace(",", "")
    
    return {
        "text": text,
        "entities": entities
    }


def generate_dataset(num_train: int = 600, num_dev: int = 150, num_test: int = 100):
    """Generate full datasets."""
    random.seed(42)
    
    datasets = {}
    
    for dataset_name, size in [("train", num_train), ("dev", num_dev), ("test", num_test)]:
        print(f"Generating {size} {dataset_name} examples...")
        data = []
        
        for i in range(size):
            utt = create_utterance()
            
            if dataset_name == "test":
                # Test set has no entities field
                utt_out = {
                    "id": f"test_{i+1:05d}",
                    "text": utt["text"]
                }
            else:
                utt_out = {
                    "id": f"{dataset_name}_{i+1:05d}",
                    "text": utt["text"],
                    "entities": utt["entities"]
                }
            
            data.append(utt_out)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{size} examples...")
        
        datasets[dataset_name] = data
        print(f"  Total {dataset_name} examples: {len(data)}\n")
    
    return datasets


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL format."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(data)} examples to {path}")


if __name__ == "__main__":
    datasets = generate_dataset(num_train=1600, num_dev=200, num_test=200)
    
    save_jsonl(datasets["train"], "data/train.jsonl")
    save_jsonl(datasets["dev"], "data/dev.jsonl")
    save_jsonl(datasets["test"], "data/test.jsonl")
    
    print("\n✓ Data generation complete!")
