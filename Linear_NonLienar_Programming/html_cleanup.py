from bs4 import BeautifulSoup
import sys

# Read the HTML file
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

# Remove unnecessary elements
if soup.find('div', class_='jp-Notebook-footer'):
    soup.find('div', class_='jp-Notebook-footer').decompose()

# Remove unused CSS classes
for tag in soup.find_all(class_=True):
    classes = tag['class']
    # Keep only essential classes
    essential_classes = [c for c in classes if c in ['output_area', 'output_text', 'output_html']]
    if essential_classes:
        tag['class'] = essential_classes
    else:
        del tag['class']

# Remove empty div containers
for div in soup.find_all('div'):
    if not div.contents or div.contents == ['\n']:
        div.decompose()

# Write the cleaned HTML back to file
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(str(soup))