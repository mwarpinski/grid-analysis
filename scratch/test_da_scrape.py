import asyncio
from datetime import datetime
from src.scrapers.spp_da_constraints_scraper import DAConstraintsScraper

async def main():
    scraper = DAConstraintsScraper()
    # Try scraping a recent day, e.g., 2026-05-15
    target_date = datetime(2026, 5, 15)
    print(f"Scraping DA constraints for {target_date.date()}...")
    df = scraper.scrape_day(target_date)
    print("Scraped DataFrame shape:", df.shape)
    if not df.empty:
        print("Columns:", df.columns.tolist())
        print(df.head())

if __name__ == "__main__":
    asyncio.run(main())
