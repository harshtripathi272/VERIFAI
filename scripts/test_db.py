from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()


url = os.getenv("SUPABASE_URL")
print(url)
key = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(url, key)

response = supabase.table("past_mistakes").select("*").limit(1).execute()
print(response)
