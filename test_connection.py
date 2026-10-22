import os
from dotenv import load_dotenv
from google.cloud import aiplatform, storage

load_dotenv()
project = os.environ["GCP_PROJECT_ID"]
region = os.environ["GCP_REGION"]
bucket_name = os.environ["GCP_BUCKET_NAME"]

aiplatform.init(project=project, location=region)
print("Vertex AI init OK :", project, region)

client = storage.Client(project=project)
blobs = list(client.list_blobs(bucket_name, max_results=3))
print("Bucket OK :", bucket_name, "-", len(blobs), "fichier(s) visibles")
