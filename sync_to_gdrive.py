import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

FOLDER_NAME = "MarketSurgeBot"
FILES_TO_SYNC = [
    "marketsurge_actionable_setups.csv",
    "marketsurge_rankings.csv",
    "marketsurge_hh_hl_setups.csv",
    "josh_brown_best_stocks.csv",
]

def main():
    creds_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("Missing GDRIVE_SERVICE_ACCOUNT_JSON environment variable.")

    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)

    # Locate the shared MarketSurgeBot folder
    folder_query = (
        f"name = '{FOLDER_NAME}' and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    folders = service.files().list(q=folder_query, fields="files(id, name)").execute().get("files", [])
    if not folders:
        raise FileNotFoundError(
            f"Folder '{FOLDER_NAME}' not found. Make sure it is shared as Editor with {creds_info.get('client_email')}."
        )
    folder_id = folders[0]["id"]

    for filename in FILES_TO_SYNC:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (not found locally).")
            continue

        media = MediaFileUpload(filename, mimetype="text/csv", resumable=True)

        # Check if file already exists in MarketSurgeBot so we overwrite in-place
        file_query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
        existing = service.files().list(q=file_query, fields="files(id, name)").execute().get("files", [])

        if existing:
            file_id = existing[0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
            print(f"Updated existing file in Google Drive: {filename}")
        else:
            metadata = {"name": filename, "parents": [folder_id]}
            service.files().create(body=metadata, media_body=media, fields="id").execute()
            print(f"Uploaded new file to Google Drive: {filename}")

if __name__ == "__main__":
    main()
