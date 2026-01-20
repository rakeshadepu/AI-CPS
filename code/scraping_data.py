import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup


def scrape_data():
    """
    Scrapes economic data from the German Federal Ministry of
    Research, Technology and Space and stores it as raw CSV data.
    """

    url = "https://www.datenportal.bmftr.bund.de/portal/en/Tabelle-1.10.2.html"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    # --------------------------------------------------
    # STEP 0: Fetch webpage
    # --------------------------------------------------
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        print(f"Error fetching data: {error}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    if table is None:
        print("No table found on page")
        return None

    # --------------------------------------------------
    # STEP 1: Parse table headers
    # --------------------------------------------------
    thead = table.find("thead")
    if thead is None:
        print("No table header found")
        return None

    header_rows = thead.find_all("tr")

    # First header row: main categories
    top_headers = []
    for th in header_rows[0].find_all("td")[1:]:
        text = re.sub(r"\d+\)", "", th.get_text(strip=True)).strip()
        axis_value = th.get("axis", "1")
        colspan = int(th.get("colspan", 1))

        combined_header = f"{text}_{axis_value}"
        top_headers.extend([combined_header] * colspan)

    # Second header row: years
    year_headers = [
        th.get_text(strip=True)
        for th in header_rows[1].find_all("td")[1:]
    ]

    # Combine category and year headers
    final_headers = []
    for category, year in zip(top_headers, year_headers):
        clean_category = (
            category.replace(" ", "")
                    .replace("(nominal)", "")
        )
        final_headers.append(f"{clean_category}_{year}")

    final_headers.insert(0, "state")

    # --------------------------------------------------
    # STEP 2: Parse table body
    # --------------------------------------------------
    tbody = table.find("tbody")
    if tbody is None:
        print("No table body found")
        return None

    rows = tbody.find_all("tr")[1:]
    data = []

    for row in rows:
        cols = row.find_all("td")
        if not cols:
            continue

        state = cols[0].get_text(strip=True)
        if not state:
            continue

        values = [
            col.get_text(strip=True).replace(",", "")
            for col in cols[1:]
        ]

        data.append([state] + values)

    # --------------------------------------------------
    # STEP 3: Create DataFrame
    # --------------------------------------------------
    df = pd.DataFrame(data, columns=final_headers)

    df.iloc[:, 1:] = df.iloc[:, 1:].apply(
        lambda series: pd.to_numeric(series.astype(str), errors="coerce")
    )

    # --------------------------------------------------
    # STEP 4: Save raw data
    # --------------------------------------------------
    os.makedirs("../data/raw", exist_ok=True)
    file_path = "../data/raw/scraped_data.csv"

    try:
        df.to_csv(file_path, index=False)
        print("\nData successfully saved:")
        print(f"  Path     : {file_path}")
        print(f"  Records  : {len(df)}")
        print(f"  Columns  : {len(df.columns)}")
        print(f"  Size     : {os.path.getsize(file_path) / 1024:.2f} KB")
        return True
    except Exception as error:
        print(f"Error saving file: {error}")
        return False


if __name__ == "__main__":
    scrape_data()
