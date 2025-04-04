from dotenv import load_dotenv
load_dotenv()
import praw
import json
import os
from datetime import datetime

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

# subreddits
SUBREDDITS = [
    "depression",
    "SuicideWatch",
    "offmychest",
    "mentalhealth"
]

# scraping
def scrape_posts(limit=500):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/raw/reddit_{timestamp}.json"
    os.makedirs("data/raw", exist_ok=True)

    all_posts = {}
    types = ["hot", "new", "top"]

    for sub in SUBREDDITS:
        print(f"Scraping r/{sub}")

        for kind in types:
            print(f"  → {kind.upper()} posts")

            if kind == "hot":
                posts = reddit.subreddit(sub).hot(limit=limit)
            elif kind == "new":
                posts = reddit.subreddit(sub).new(limit=limit)
            elif kind == "top":
                posts = reddit.subreddit(sub).top(limit=limit)

            for post in posts:
                if post.id not in all_posts:
                    all_posts[post.id] = {
                        "id": post.id,
                        "subreddit": sub,
                        "title": post.title,
                        "selftext": post.selftext,
                        "created_utc": post.created_utc,
                        "score": post.score,
                        "url": post.url
                    }

    # saving to json
    with open(output_path, "w") as f:
        json.dump(list(all_posts.values()), f, indent=2)

    print(f"\nSaved {len(all_posts)} unique posts to {output_path}")

if __name__ == "__main__":
    scrape_posts(limit=500)
