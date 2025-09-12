from pydrive2.auth import ServiceAccountCredentials
from pydrive2.drive import GoogleDrive
from pydrive2.settings import LoadSettingsFile
from typing import List, Dict

from config import settings

SCOPES = ['https://www.googleapis.com/auth/drive.readonly',
          'https://www.googleapis.com/auth/spreadsheets.readonly']

def make_drive_client() -> GoogleDrive:
    # Autenticação via Service Account
    gauth = LoadSettingsFile({})
    gauth.service_config = {
        'client_config_backend': 'service',
        'service_config': {
            'client_json_file_path': settings.CREDENTIALS_JSON
        }
    }
    creds = ServiceAccountCredentials.from_json_keyfile_name(settings.CREDENTIALS_JSON, SCOPES)
    gauth.credentials = creds
    return GoogleDrive(gauth)

def list_files_in_folder(drive: GoogleDrive, folder_id: str) -> List[Dict]:
    # Lista arquivos (recursivo em subpastas)
    q = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': q}).GetList()
    results = []
    for f in file_list:
        if f['mimeType'] == 'application/vnd.google-apps.folder':
            results.extend(list_files_in_folder(drive, f['id']))
        else:
            results.append(f)
    return results

def download_file(drive: GoogleDrive, file_meta: Dict, dst_path: str):
    f = drive.CreateFile({'id': file_meta['id']})
    f.FetchMetadata()

    mime = f['mimeType']
    export_map = {
        'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    }
    if mime in export_map:
        f.GetContentFile(dst_path, mimetype=export_map[mime])
    else:
        f.GetContentFile(dst_path)
