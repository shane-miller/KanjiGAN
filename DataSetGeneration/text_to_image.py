from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
from tqdm import tqdm
import pathlib
import shutil
import os

#Make folder to save images
current_file_path = pathlib.Path(__file__).parent.absolute()
path = current_file_path.parents[0] / 'input_images' / 'images'
shutil.rmtree(path, ignore_errors=True)
os.mkdir(path)

# Open kanji list
file = open(current_file_path / 'kanji.txt', 'r', encoding='utf8')

# Create list of lines
lines = file.readlines()

# Loop through each kanji
for char_num, char in enumerate(tqdm(lines, desc="Generating Images"), 1):
    # Create image
    img = Image.new('RGB', (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(current_file_path / 'NotoSansJP-Regular.otf'), 225)
    draw.text((15, -50), char.strip(), fill='black', font=font, align='center')

    # Save image
    img.save(path / (str(char_num) + '.png'), 'PNG')
