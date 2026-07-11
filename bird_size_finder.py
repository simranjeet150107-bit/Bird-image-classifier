"""
bird_size_finder.py
--------------------
Enter a bird's body length and wingspan (in cm) and get the closest
matching species — name, size range, and its photo (fetched live from
Wikipedia).

Fully type-annotated Python (uses `typing` + `dataclasses`).

Run in Google Colab or Jupyter for inline images, or in a normal
terminal (it will just print the image URL instead of showing it).
"""

from dataclasses import dataclass
from typing import Optional, List
import requests
from io import BytesIO

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    CAN_SHOW_IMAGE = True
except ImportError:
    CAN_SHOW_IMAGE = False


# ---------- Data model ----------

@dataclass
class Bird:
    name: str
    scientific_name: str
    length_min_cm: float
    length_max_cm: float
    wingspan_min_cm: float
    wingspan_max_cm: float
    wiki_title: str  # Wikipedia article title, used to fetch a live photo


@dataclass
class MatchResult:
    bird: Bird
    score: float  # 0.0 = perfect match (within both ranges)


# ---------- Database (~30 species, small to large) ----------

BIRDS: List[Bird] = [
    Bird("Bee Hummingbird", "Mellisuga helenae", 5, 6, 6, 7, "Bee_hummingbird"),
    Bird("Ruby-throated Hummingbird", "Archilochus colubris", 7, 9, 8, 11, "Ruby-throated_hummingbird"),
    Bird("Purple Sunbird", "Cinnyris asiaticus", 10, 11, 12, 15, "Purple_sunbird"),
    Bird("European Robin", "Erithacus rubecula", 12.5, 14, 20, 22, "European_robin"),
    Bird("House Sparrow", "Passer domesticus", 14, 16, 21, 25, "House_sparrow"),
    Bird("Common Kingfisher", "Alcedo atthis", 16, 18, 24, 26, "Common_kingfisher"),
    Bird("Indian Robin", "Copsychus fulicatus", 16, 19, 21, 25, "Indian_robin"),
    Bird("Green Bee-eater", "Merops orientalis", 16, 20, 28, 30, "Green_bee-eater"),
    Bird("Red-vented Bulbul", "Pycnonotus cafer", 20, 22, 28, 32, "Red-vented_bulbul"),
    Bird("Northern Cardinal", "Cardinalis cardinalis", 21, 23, 25, 31, "Northern_cardinal"),
    Bird("Common Myna", "Acridotheres tristis", 23, 26, 40, 45, "Common_myna"),
    Bird("Jungle Babbler", "Argya striata", 23, 25, 32, 35, "Jungle_babbler"),
    Bird("Laughing Dove", "Spilopelia senegalensis", 25, 27, 34, 40, "Laughing_dove"),
    Bird("Hoopoe", "Upupa epops", 25, 32, 44, 48, "Hoopoe"),
    Bird("Spotted Dove", "Spilopelia chinensis", 28, 30, 40, 45, "Spotted_dove"),
    Bird("White-throated Kingfisher", "Halcyon smyrnensis", 27, 28, 44, 46, "White-throated_kingfisher"),
    Bird("Black Drongo", "Dicrurus macrocercus", 28, 32, 40, 45, "Black_drongo"),
    Bird("Eurasian Collared Dove", "Streptopelia decaocto", 30, 33, 47, 55, "Eurasian_collared_dove"),
    Bird("Indian Roller", "Coracias benghalensis", 30, 34, 65, 74, "Indian_roller"),
    Bird("Rose-ringed Parakeet", "Psittacula krameri", 38, 42, 42, 48, "Rose-ringed_parakeet"),
    Bird("Asian Koel", "Eudynamys scolopaceus", 39, 46, 60, 65, "Asian_koel"),
    Bird("Indian Pond Heron", "Ardeola grayii", 42, 45, 80, 90, "Indian_pond_heron"),
    Bird("House Crow", "Corvus splendens", 40, 43, 76, 85, "House_crow"),
    Bird("Cattle Egret", "Bubulcus ibis", 46, 56, 88, 96, "Cattle_egret"),
    Bird("Barn Owl", "Tyto alba", 33, 39, 80, 95, "Barn_owl"),
    Bird("Mallard", "Anas platyrhynchos", 50, 65, 81, 98, "Mallard"),
    Bird("Black Kite", "Milvus migrans", 55, 60, 120, 140, "Black_kite"),
    Bird("Grey Heron", "Ardea cinerea", 84, 102, 155, 175, "Grey_heron"),
    Bird("Indian Peafowl", "Pavo cristatus", 100, 115, 140, 160, "Indian_peafowl"),
    Bird("Golden Eagle", "Aquila chrysaetos", 66, 100, 180, 220, "Golden_eagle"),
    Bird("Bald Eagle", "Haliaeetus leucocephalus", 70, 90, 180, 230, "Bald_eagle"),
    Bird("Mute Swan", "Cygnus olor", 125, 160, 200, 240, "Mute_swan"),
    Bird("Sarus Crane", "Antigone antigone", 152, 176, 220, 250, "Sarus_crane"),
    Bird("Ostrich", "Struthio camelus", 210, 280, 180, 200, "Common_ostrich"),
]


# ---------- Matching logic ----------

def _range_distance(value: float, low: float, high: float) -> float:
    """0 if value is inside [low, high], else the gap to the nearest edge."""
    if low <= value <= high:
        return 0.0
    return low - value if value < low else value - high


def _score(bird: Bird, length_cm: float, wingspan_cm: float) -> float:
    length_span = max(bird.length_max_cm - bird.length_min_cm, 1.0)
    wingspan_span = max(bird.wingspan_max_cm - bird.wingspan_min_cm, 1.0)
    length_dist = _range_distance(length_cm, bird.length_min_cm, bird.length_max_cm) / length_span
    wingspan_dist = _range_distance(wingspan_cm, bird.wingspan_min_cm, bird.wingspan_max_cm) / wingspan_span
    return length_dist + wingspan_dist


def find_best_matches(length_cm: float, wingspan_cm: float, top_n: int = 3) -> List[MatchResult]:
    """Return the top_n closest species, sorted best match first."""
    results = [MatchResult(bird, _score(bird, length_cm, wingspan_cm)) for bird in BIRDS]
    results.sort(key=lambda r: r.score)
    return results[:top_n]


# ---------- Image fetching ----------

def fetch_wikipedia_image_url(wiki_title: str) -> Optional[str]:
    """Fetch a live thumbnail image URL for a species from Wikipedia."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": wiki_title,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": 400,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        thumbnail = page.get("thumbnail", {})
        return thumbnail.get("source")
    except requests.RequestException:
        return None


def show_image(image_url: str, title: str) -> None:
    """Display an image inline (Colab/Jupyter) or print the URL (terminal)."""
    if CAN_SHOW_IMAGE:
        try:
            resp = requests.get(image_url, timeout=8)
            img = Image.open(BytesIO(resp.content))
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.title(title)
            plt.show()
            return
        except Exception:
            pass
    print(f"Photo: {image_url}")


# ---------- Main lookup function ----------

def identify_bird(length_cm: float, wingspan_cm: float, show_photo: bool = True) -> None:
    """Print (and optionally show) the best-matching bird for the given size."""
    matches = find_best_matches(length_cm, wingspan_cm, top_n=3)
    best = matches[0]
    b = best.bird

    confidence = "Exact range match" if best.score == 0 else f"Closest match (score {best.score:.2f})"

    print("=" * 50)
    print(f"  {confidence}")
    print(f"  Name: {b.name}")
    print(f"  Scientific name: {b.scientific_name}")
    print(f"  Typical length: {b.length_min_cm}-{b.length_max_cm} cm")
    print(f"  Typical wingspan: {b.wingspan_min_cm}-{b.wingspan_max_cm} cm")
    print(f"  You entered: {length_cm} cm / {wingspan_cm} cm")
    print("=" * 50)

    if len(matches) > 1:
        print("\nAlso close in size:")
        for m in matches[1:]:
            print(f"  - {m.bird.name} ({m.bird.length_min_cm}-{m.bird.length_max_cm}cm / "
                  f"{m.bird.wingspan_min_cm}-{m.bird.wingspan_max_cm}cm)")

    if show_photo:
        image_url = fetch_wikipedia_image_url(b.wiki_title)
        if image_url:
            show_image(image_url, b.name)
        else:
            print(f"\n(No photo found — see https://en.wikipedia.org/wiki/{b.wiki_title})")


# ---------- CLI entry point ----------

if __name__ == "__main__":
    print("Bird Size Identifier — enter measurements in centimetres.\n")
    try:
        length_input: float = float(input("Body length (cm): "))
        wingspan_input: float = float(input("Wingspan (cm): "))
    except ValueError:
        print("Please enter valid numbers.")
    else:
        identify_bird(length_input, wingspan_input)
