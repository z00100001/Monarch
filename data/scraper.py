from dotenv import load_dotenv
load_dotenv()
import praw
import json
import os
from datetime import datetime

# creds
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
    user_agent=REDDIT_USER_AGENT
)


SUBREDDITS = [
    "depression",
    "SuicideWatch",
    "offmychest",
    "mentalhealth"
]

# scraper
def scrape_posts(limit=100):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/raw/reddit_{timestamp}.json"
    os.makedirs("data/raw", exist_ok=True)

    all_posts = []

    for sub in SUBREDDITS:
        print(f"Scraping r/{sub}...")
        for post in reddit.subreddit(sub).hot(limit=limit):
            all_posts.append({
                "id": post.id,
                "subreddit": sub,
                "title": post.title,
                "selftext": post.selftext,
                "created_utc": post.created_utc,
                "score": post.score,
                "url": post.url
            })

    with open(output_path, "w") as f:
        json.dump(all_posts, f, indent=2)

    print(f"\nSaved {len(all_posts)} posts to {output_path}")

if __name__ == "__main__":
    scrape_posts(limit=200)
