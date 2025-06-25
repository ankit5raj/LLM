from langchain.text_splitter import RecursiveCharacterTextSplitter, Language



code = '''
# sample_code.py

import math

def calculate_area(radius):
    """Calculate the area of a circle."""
    if radius < 0:
        raise ValueError("Radius cannot be negative.")
    return math.pi * radius ** 2

def print_area_table():
    """Print areas for radii from 1 to 5."""
    for r in range(1, 6):
        area = calculate_area(r)
        print(f"Radius: {r}, Area: {area:.2f}")

class Circle:
    """Class representing a circle."""

    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return calculate_area(self.radius)
    
    def describe(self):
        return f"A circle with radius {self.radius} has area {self.area():.2f}"

if __name__ == "__main__":
    print_area_table()
    c = Circle(3)
    print(c.describe())

'''


splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size = 200,
    chunk_overlap = 0
)

result = splitter.split_text(code)

print(result[2])