# -*- coding: utf-8 -*-

# Sample Python code for youtube.commentThreads.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
### pip install --upgrade google-api-python-client
import googleapiclient.discovery
import googleapiclient.errors

### pip install --upgrade google-api-python-client
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('YOUTUBE_API')


scopes = ["https://www.googleapis.com/auth/youtube.readonly"]

def channelList():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        forUsername="GoogleDevelopers"  ## List name of channels you want info of
    )
    response = request.execute()

    print(response)

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = api_key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId="-mAG0Tp87S4"
    )
    response = request.execute()

    print(response)

if __name__ == "__main__":
    main()