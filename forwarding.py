import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import json
from typing import Optional

def get_firebase_config():
    firebase_key = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
    if firebase_key:
        try:
            return json.loads(firebase_key)
        except json.JSONDecodeError as e:
            print(f"Error parsing FIREBASE_SERVICE_ACCOUNT_KEY: {str(e)}")
            raise

    google_creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if google_creds_path and os.path.exists(google_creds_path):
        return google_creds_path
        
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        raise ValueError(
            "Firebase configuration not found. Please set one of:\n"
            "1. FIREBASE_SERVICE_ACCOUNT_KEY (JSON string of service account key)\n"
            "2. GOOGLE_APPLICATION_CREDENTIALS (path to service account key file)\n"
            "3. Ensure GOOGLE_CLOUD_PROJECT is set for project identification"
        )
    
    raise ValueError(
        "Could not load Firebase credentials. Please check:\n"
        f"- FIREBASE_SERVICE_ACCOUNT_KEY is valid JSON\n"
        f"- GOOGLE_APPLICATION_CREDENTIALS points to existing file: {google_creds_path}\n"
        f"- GOOGLE_CLOUD_PROJECT is set: {project_id}"
    )

def initialize_firebase():
    if firebase_admin._apps:
        print("Firebase already initialized, using existing app")
        return firestore.client()
    
    try:
        firebase_config = get_firebase_config()
        
        if isinstance(firebase_config, str):
            cred = credentials.Certificate(firebase_config)
        else:
            cred = credentials.Certificate(firebase_config)
       
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if project_id:
            firebase_admin.initialize_app(cred, {'projectId': project_id})
        else:
            firebase_admin.initialize_app(cred)
        
        print("Firebase initialized successfully")
        return firestore.client()
        
    except Exception as e:
        print(f"❌ Error initializing Firebase: {str(e)}")
        raise

def get_db():
    return initialize_firebase()
db = get_db()

def save_unanswered_question(question_english):
    try:
        doctor_doc_ref = db.collection("DOCTOR").document("1")
        doc = doctor_doc_ref.get()
        data = doc.to_dict() if doc.exists else {}
        qn_list = data.get("qn", [])
        if question_english not in qn_list:
            qn_list.append(question_english)
            doctor_doc_ref.set({
                "qn": qn_list
            }, merge=True)
            
            print(f"✅ Unanswered question saved for doctor review: {question_english}")
        else:
            print(f"ℹ️  Question already exists in unanswered list: {question_english}")
            
    except Exception as e:
        print(f"❌ Error saving unanswered question: {str(e)}")
        raise e

def save_user_interaction(question_english, answer_english, user_session_id=None):
    try:
        interaction_data = {
            "question": question_english,
            "answer": answer_english,
            "timestamp": datetime.now(),
            "session_id": user_session_id or "anonymous",
            "status": "answered"
        }
        
        unanswered_indicators = [
            "doctor has been notified",
            "doctor will be notified", 
            "check back in a few days",
            "unable to answer your question"
        ]
        
        if any(indicator in answer_english.lower() for indicator in unanswered_indicators):
            interaction_data["status"] = "forwarded_to_doctor"

        db.collection("user").add(interaction_data)
        
        print(f"✅ User interaction saved: Q: {question_english[:50]}...")
        
    except Exception as e:
        print(f"❌ Error saving user interaction: {str(e)}")
        raise e
