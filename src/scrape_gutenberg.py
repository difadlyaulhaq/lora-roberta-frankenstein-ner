import subprocess
import os
import json
import re
from bs4 import BeautifulSoup

def download_file(url, output_path):
    print(f"Downloading {url}...")
    try:
        # Using curl.exe directly as it proved more reliable in this environment
        subprocess.run([
            "curl.exe", "-L", "-o", output_path, url,
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return False

def scrape_frankenstein(file_path):
    print(f"Reading data from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Extract Metadata
    title_raw = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Frankenstein"
    title = title_raw.rstrip(';').strip()
    
    author = "Mary Wollstonecraft Shelley"
    author_tag = soup.find('h2', string=re.compile(r'Shelley', re.I))
    if author_tag:
        author = author_tag.get_text(strip=True).replace("by ", "").strip()

    print(f"Title: {title}")
    print(f"Author: {author}")

    content_data = {
        "metadata": {
            "title": title,
            "author": author,
            "source": "https://www.gutenberg.org/ebooks/84.html.images"
        },
        "chapters": []
    }

    # 2. Extract Content
    headings = soup.find_all(['h2', 'h3'])
    
    current_chapter = None
    current_text = []

    for heading in headings:
        header_text = heading.get_text(strip=True)
        
        # Skip title/author repetitions and standard Gutenberg headers
        skip_keywords = ["CONTENTS", "PROJECT GUTENBERG", "TRANSCRIPTION NOTES", "PROMETHEUS", "SHELLEY"]
        if any(skip in header_text.upper() for skip in skip_keywords):
            continue
            
        if current_chapter:
            content_data["chapters"].append({
                "title": current_chapter,
                "content": "\n".join(current_text).strip()
            })
            current_text = []

        current_chapter = header_text
        
        next_node = heading.find_next_sibling()
        while next_node and next_node.name not in ['h2', 'h3']:
            if next_node.name == 'p':
                text = next_node.get_text(strip=True)
                if text:
                    current_text.append(text)
            next_node = next_node.find_next_sibling()

    if current_chapter:
        content_data["chapters"].append({
            "title": current_chapter,
            "content": "\n".join(current_text).strip()
        })

    # Save to directory of file_path
    base_dir = os.path.dirname(file_path)

    # 3. Save to JSON
    json_output = os.path.join(base_dir, "frankenstein_data.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(content_data, f, indent=4, ensure_ascii=False)
    
    # 4. Save to CSV (Useful for Pandas/ML datasets)
    import csv
    csv_output = os.path.join(base_dir, "frankenstein_dataset.csv")
    with open(csv_output, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["title", "content"])
        for chapter in content_data["chapters"]:
            writer.writerow([chapter["title"], chapter["content"]])

    # 5. Save to TXT (Clean plain text for training)
    txt_output = os.path.join(base_dir, "frankenstein_clean.txt")
    with open(txt_output, "w", encoding="utf-8") as f:
        for chapter in content_data["chapters"]:
            f.write(f"--- {chapter['title']} ---\n")
            f.write(chapter["content"] + "\n\n")

    print(f"\nSuccess! Dataset created in multiple formats:")
    print(f"- JSON: {json_output}")
    print(f"- CSV:  {csv_output}")
    print(f"- TXT:  {txt_output}")
    print(f"Total chapters/sections processed: {len(content_data['chapters'])}")

if __name__ == "__main__":
    target_url = "https://www.gutenberg.org/ebooks/84.html.images"
    # Locate data folder relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    local_file = os.path.join(data_dir, "frankenstein.html")
    
    if not os.path.exists(local_file):
        if download_file(target_url, local_file):
            scrape_frankenstein(local_file)
    else:
        scrape_frankenstein(local_file)
