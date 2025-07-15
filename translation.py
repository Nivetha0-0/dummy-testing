import os
import json
import tempfile
import streamlit as st
from google.cloud import translate
from google.cloud import texttospeech
from google.cloud import speech
from typing import Optional, List

# Global clients
_translator_client = None
_texttospeech_client = None
_speech_client = None

def _initialize_gc_client(client_class) -> Optional:
    try:
        creds = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
        if isinstance(creds, str):
            creds = json.loads(creds)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(creds, temp_file)
            temp_credentials_path = temp_file.name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path
        client = client_class()
        os.unlink(temp_credentials_path)
        return client

    except Exception as e:
        print(f"Error initializing Google Cloud {client_class.__name__}: {e}")
        return None

def get_translator_client() -> Optional[translate.TranslationServiceClient]:
    global _translator_client
    if _translator_client is None:
        _translator_client = _initialize_gc_client(translate.TranslationServiceClient)
    return _translator_client

def get_texttospeech_client() -> Optional[texttospeech.TextToSpeechClient]:
    global _texttospeech_client
    if _texttospeech_client is None:
        _texttospeech_client = _initialize_gc_client(texttospeech.TextToSpeechClient)
    return _texttospeech_client

def get_speech_client() -> Optional[speech.SpeechClient]:
    global _speech_client
    if _speech_client is None:
        _speech_client = _initialize_gc_client(speech.SpeechClient)
    return _speech_client

def get_supported_languages(client, allowed_langs: Optional[List[str]] = None) -> dict[str, str]:
    if not client:
        return {}
    try:
        parent = f"projects/{st.secrets['GOOGLE_CLOUD_PROJECT']}/locations/global"
        response = client.get_supported_languages(parent=parent, display_language_code='en')
        return {
            lang.language_code: lang.display_name or lang.language_code
            for lang in response.languages
            if not allowed_langs or lang.language_code in allowed_langs
        }
    except Exception as e:
        print(f"Error fetching supported languages: {e}")
        return {}

def translate_text(client, text: Optional[str], target_language_code: str, source_language_code: str) -> Optional[str]:
    if not text or not client or source_language_code == target_language_code:
        return text
    try:
        parent = f"projects/{st.secrets['GOOGLE_CLOUD_PROJECT']}/locations/global"
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": source_language_code,
                "target_language_code": target_language_code,
            }
        )
        return response.translations[0].translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return None

# Helper functions to access secrets
def get_openai_key() -> str:
    return st.secrets["OPENAI_KEY"]

def get_mongodb_uri() -> str:
    return st.secrets["MONGODB_URI"]

def get_firebase_service_account_key():
    key = st.secrets["FIREBASE_SERVICE_ACCOUNT_KEY"]
    return json.loads(key) if isinstance(key, str) else key

def get_google_cloud_project() -> str:
    return st.secrets["GOOGLE_CLOUD_PROJECT"]
