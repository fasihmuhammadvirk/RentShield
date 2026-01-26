"""
Dataset builder for RentShield Germany.
Generates synthetic training data for demonstration.
In production, use real German rental data from ImmoScout24 or similar.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path


# German cities with typical rent ranges (€/m²)
CITIES_DATA = {
    "Munich": {"min_rent": 16, "max_rent": 28, "tier": 1},
    "Frankfurt": {"min_rent": 14, "max_rent": 24, "tier": 1},
    "Stuttgart": {"min_rent": 13, "max_rent": 22, "tier": 1},
    "Berlin": {"min_rent": 11, "max_rent": 20, "tier": 2},
    "Hamburg": {"min_rent": 12, "max_rent": 20, "tier": 2},
    "Düsseldorf": {"min_rent": 11, "max_rent": 18, "tier": 2},
    "Cologne": {"min_rent": 10, "max_rent": 17, "tier": 2},
    "Hannover": {"min_rent": 8, "max_rent": 14, "tier": 3},
    "Dresden": {"min_rent": 7, "max_rent": 12, "tier": 3},
    "Leipzig": {"min_rent": 6, "max_rent": 11, "tier": 3},
    "Nuremberg": {"min_rent": 9, "max_rent": 15, "tier": 3},
    "Essen": {"min_rent": 6, "max_rent": 10, "tier": 4},
    "Dortmund": {"min_rent": 6, "max_rent": 10, "tier": 4},
    "Duisburg": {"min_rent": 5, "max_rent": 9, "tier": 4},
}

# Postcodes by city (sample ranges)
CITY_POSTCODES = {
    "Munich": ["80331", "80333", "80335", "80469", "80539", "81667", "81669"],
    "Frankfurt": ["60311", "60313", "60316", "60318", "60322", "60325"],
    "Stuttgart": ["70173", "70178", "70180", "70182", "70186", "70188"],
    "Berlin": ["10115", "10117", "10119", "10178", "10179", "10243", "10963"],
    "Hamburg": ["20095", "20097", "20099", "20144", "20146", "20148"],
    "Düsseldorf": ["40210", "40211", "40213", "40215", "40217", "40219"],
    "Cologne": ["50667", "50668", "50670", "50672", "50674", "50676"],
    "Hannover": ["30159", "30161", "30163", "30165", "30167", "30169"],
    "Dresden": ["01067", "01069", "01097", "01099", "01109", "01127"],
    "Leipzig": ["04103", "04105", "04107", "04109", "04177", "04179"],
    "Nuremberg": ["90402", "90403", "90408", "90409", "90419", "90429"],
    "Essen": ["45127", "45128", "45130", "45131", "45133", "45134"],
    "Dortmund": ["44135", "44137", "44139", "44141", "44143", "44145"],
    "Duisburg": ["47051", "47053", "47055", "47057", "47058", "47059"],
}

# Normal listing descriptions (German/English mix)
NORMAL_DESCRIPTIONS = [
    "Schöne helle Wohnung im Altbau mit Balkon. Ruhige Lage.",
    "Modern renovated apartment near public transport.",
    "Gemütliche 2-Zimmer Wohnung mit Einbauküche.",
    "Bright apartment with great views of the city.",
    "Zentrale Lage, alle Einkaufsmöglichkeiten fußläufig erreichbar.",
    "Newly built apartment with underground parking.",
    "Altbauwohnung mit hohen Decken und Parkettboden.",
    "Quiet neighborhood, perfect for families.",
    "Moderne Wohnung mit Fußbodenheizung.",
    "Well-maintained building with elevator.",
    "Sonnige Südausrichtung mit großer Terrasse.",
    "Close to university and city center.",
    "Frisch saniert mit neuer Küche und Bad.",
    "Green surroundings with garden access.",
    "Gepflegtes Mehrfamilienhaus in guter Lage.",
]

# Scam-like descriptions
SCAM_DESCRIPTIONS = [
    "I am currently abroad, please send deposit via Western Union before viewing.",
    "Sehr günstig wegen Umzug ins Ausland. Kaution vorab überweisen.",
    "URGENT: Must rent immediately. Wire transfer required before keys.",
    "Cannot meet in person, will send keys by mail after payment.",
    "Too good to be true? It's real! Just send deposit first.",
    "Ich bin im Ausland, bitte Miete vorab überweisen für Schlüsselübergabe.",
    "My company is relocating me abroad. Send first month rent to reserve.",
    "Keine Besichtigung möglich. Schlüssel werden nach Zahlung geschickt.",
    "Payment required before viewing. Keys will be mailed to you.",
    "I won the lottery and leaving Germany. Quick decision needed.",
]


def generate_listing(is_scam: bool = False, overpriced: bool = False) -> dict:
    """Generate a single rental listing."""
    
    city = random.choice(list(CITIES_DATA.keys()))
    city_data = CITIES_DATA[city]
    postcode = random.choice(CITY_POSTCODES[city])
    
    # Generate realistic apartment features
    living_space = round(random.uniform(25, 150), 1)
    rooms = max(1, round(living_space / 30 + random.uniform(-0.5, 0.5), 1))
    year_built = random.randint(1900, 2024)
    floor = random.randint(0, 8)
    
    # Base rent calculation
    base_rent_per_sqm = random.uniform(city_data["min_rent"], city_data["max_rent"])
    
    # Adjustments based on features
    if year_built > 2015:
        base_rent_per_sqm *= 1.1
    elif year_built < 1960:
        base_rent_per_sqm *= 0.9
    
    if floor >= 4:
        base_rent_per_sqm *= 1.05
    elif floor == 0:
        base_rent_per_sqm *= 0.95
    
    if living_space > 100:
        base_rent_per_sqm *= 0.92
    
    rent = round(base_rent_per_sqm * living_space, 0)
    
    # Apply modifications for scam/overpriced scenarios
    if is_scam:
        # Scams often have unrealistically low prices
        rent = rent * random.uniform(0.3, 0.6)
        description = random.choice(SCAM_DESCRIPTIONS)
        scam_label = 1
    else:
        description = random.choice(NORMAL_DESCRIPTIONS)
        scam_label = 0
    
    if overpriced:
        rent = rent * random.uniform(1.3, 1.8)
    
    return {
        "city": city,
        "postcode": postcode,
        "living_space": living_space,
        "rooms": rooms,
        "rent": round(rent, 0),
        "description": description,
        "year_built": year_built,
        "floor": floor,
        "rent_per_sqm": round(rent / living_space, 2),
        "scam_label": scam_label,
        "city_tier": city_data["tier"]
    }


def build_training_dataset(
    n_samples: int = 5000,
    scam_ratio: float = 0.15,
    overpriced_ratio: float = 0.20
) -> pd.DataFrame:
    """
    Build training dataset with realistic distribution.
    
    Args:
        n_samples: Total number of samples
        scam_ratio: Ratio of scam listings
        overpriced_ratio: Ratio of overpriced listings
    
    Returns:
        pandas DataFrame with training data
    """
    listings = []
    
    n_scams = int(n_samples * scam_ratio)
    n_overpriced = int(n_samples * overpriced_ratio)
    n_normal = n_samples - n_scams - n_overpriced
    
    print(f"Generating {n_normal} normal listings...")
    for _ in range(n_normal):
        listings.append(generate_listing(is_scam=False, overpriced=False))
    
    print(f"Generating {n_overpriced} overpriced listings...")
    for _ in range(n_overpriced):
        listings.append(generate_listing(is_scam=False, overpriced=True))
    
    print(f"Generating {n_scams} scam listings...")
    for _ in range(n_scams):
        listings.append(generate_listing(is_scam=True, overpriced=False))
    
    df = pd.DataFrame(listings)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def build_sample_dataset(n_samples: int = 50) -> pd.DataFrame:
    """Build a small sample dataset for testing."""
    return build_training_dataset(
        n_samples=n_samples,
        scam_ratio=0.2,
        overpriced_ratio=0.2
    )


if __name__ == "__main__":
    # Build datasets
    print("=" * 50)
    print("RentShield - Dataset Builder")
    print("=" * 50)
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    sample_dir = data_dir / "sample"
    training_dir = data_dir / "training"
    
    # Create directories
    sample_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample dataset
    print("\n📊 Generating sample dataset...")
    sample_df = build_sample_dataset(50)
    sample_path = sample_dir / "sample_listings.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"✓ Saved {len(sample_df)} samples to {sample_path}")
    
    # Generate training dataset
    print("\n📊 Generating training dataset...")
    training_df = build_training_dataset(5000)
    training_path = training_dir / "training_data.csv"
    training_df.to_csv(training_path, index=False)
    print(f"✓ Saved {len(training_df)} samples to {training_path}")
    
    # Print statistics
    print("\n📈 Dataset Statistics:")
    print(f"  Training samples: {len(training_df)}")
    print(f"  Scam listings: {training_df['scam_label'].sum()} ({training_df['scam_label'].mean()*100:.1f}%)")
    print(f"  Cities: {training_df['city'].nunique()}")
    print(f"  Avg rent: €{training_df['rent'].mean():.0f}")
    print(f"  Avg rent/m²: €{training_df['rent_per_sqm'].mean():.2f}")
    
    print("\n✅ Dataset generation complete!")
