import praw
import os
import pandas as pd
import json

with open('reddit_data/keys.json', encoding='utf-8') as f:
    reddit_keys = json.loads(f.read())



reddit = praw.Reddit(
    **reddit_keys
)

subreddit = reddit.subreddit('Polska')
submissions = []


for submission in subreddit.hot(limit=None):
    flair = submission.link_flair_text
    if flair == 'Polityka':
        submissions.append(submission)

print(f'Fetched {len(submissions)} submissions with flair "Polityka"')

ids = []
comments = []

for submission in submissions:
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        ids.append(comment.id)
        comments.append(comment.body)

df = pd.DataFrame({'id': ids, 'comment': comments})
df.to_csv('reddit_data\data\data.csv')

