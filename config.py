import os


global CHAT_RATE_LIMIT

CHAT_RATE_LIMIT = int(os.getenv("CHAT_RATE_LIMIT", 2))