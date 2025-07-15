import streamlit as st
from streamlit_chat import message
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, SecretStr

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pymongo.mongo_client import MongoClient

from google.cloud import translate, texttospeech, speech
from translation import (
    get_translator_client, get_texttospeech_client, get_speech_client,
    translate_text, get_supported_languages
)

# Initialize Google Cloud clients
translator_client: Optional[translate.TranslationServiceClient] = get_translator_client()
if not translator_client:
    st.warning("Error with Google Cloud Translator client")

texttospeech_client: Optional[texttospeech.TextToSpeechClient] = get_texttospeech_client()
if not texttospeech_client:
    st.warning("Error with Google Cloud Text-to-Speech")

speech_client: Optional[speech.SpeechClient] = get_speech_client()
if not speech_client:
    st.warning("Google Cloud Speech-to-Text client could not be initialized.")

ALLOWED_LANGUAGES: List[str] = ['en', 'ta', 'te', 'hi']
DEFAULT_LANGUAGE: Literal['en'] = "en"
SUPPORTED_LANGUAGES: dict[str, str] = get_supported_languages(translator_client, allowed_langs=ALLOWED_LANGUAGES) or {'en': 'English'}

tagging_prompt = ChatPromptTemplate.from_template("""
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
""")

class CasualSubject(BaseModel):
    description: str = Field(description="Classify the given user query into 'Casual Greeting' or 'Subject-Specific'")
    category: Literal['Casual Greeting', 'Subject-Specific']

class RelatedNot(BaseModel):
    description: str = Field(description="Determine whether the query is related to animal bites.")
    category: Literal['Animal Bite-Related', 'Not Animal Bite-Related']

openai_api_key_secret = SecretStr(st.secrets["OPENAI_KEY"])
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key_secret)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=openai_api_key_secret)
smaller_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=openai_api_key_secret)
larger_llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key_secret)

try:
    client = MongoClient(st.secrets["MONGODB_URI"])
    db = client["pdf_file"]
    collection = db["animal_bites"]
    _ = db.list_collection_names()
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_language" not in st.session_state:
    st.session_state.selected_language = DEFAULT_LANGUAGE

def process_input():
    user_input_original = st.session_state.user_input.strip()
    if not user_input_original:
        return

    current_lang = st.session_state.selected_language or DEFAULT_LANGUAGE
    user_input_english = translate_text(translator_client, user_input_original, DEFAULT_LANGUAGE, current_lang) or user_input_original

    retrieval_prompt_template = f"""Given a chat_history and the latest_user_input question/statement \
which MIGHT reference context in the chat history, formulate a standalone question/statement. Do NOT answer the question. \
chat_history: {st.session_state.chat_history}
latest_user_input: {user_input_english}"""

    modified_user_input = larger_llm.invoke(retrieval_prompt_template).content or user_input_english

    classification_category = 'Subject-Specific'
    try:
        response = smaller_llm.with_structured_output(CasualSubject).invoke(
            tagging_prompt.invoke({"input": modified_user_input}))
        classification_category = response.category
    except Exception as e:
        st.error(f"Error classifying query type: {e}")

    bot_response_english = ""

    if classification_category == 'Subject-Specific':
        try:
            embedding = embeddings_model.embed_query(modified_user_input)
            result = collection.aggregate([{
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embeddings",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": 3
                }
            }])

            context = ""
            for doc in result:
                db_embedding = doc["embeddings"]
                similarity = cosine_similarity([db_embedding], [embedding])[0][0]
                if round(similarity, 2) >= 0.55:
                    context += doc["raw_data"] + "\n\n"

            if context.strip():
                prompt_template = f"""You are a chatbot meant to answer questions related to animal bites. Answer the question based on the given context.
context: {context}
question: {modified_user_input}"""
                bot_response_english = llm.invoke(prompt_template).content
            else:
                relevance_category = 'Animal Bite-Related'
                try:
                    response_related_not = smaller_llm.with_structured_output(RelatedNot).invoke(
                        tagging_prompt.invoke({"input": modified_user_input}))
                    relevance_category = response_related_not.category
                except Exception as e:
                    st.error(f"Error classifying relevance: {e}")

                if relevance_category == 'Not Animal Bite-Related':
                    bot_response_english = "Sorry, I specialize in questions related to animal bites. If you have one, I'm happy to help!"
                else:
                    bot_response_english = "I am unable to answer your question at the moment. The Doctor has been notified. Please check back later."
                    try:
                        from forwarding import save_unanswered_question
                        save_unanswered_question(user_input_english)
                    except Exception as e:
                        st.error(f"Error forwarding question: {e}")
        except Exception as e:
            st.error(f"Error processing question: {e}")
            bot_response_english = "An error occurred. Please try again."
    else:
        try:
            response = llm.invoke(f"system:You are a friendly chatbot that answers questions about animal bites.\nquestion: {user_input_english}")
            bot_response_english = response.content
        except Exception as e:
            st.error(f"Error generating greeting: {e}")
            bot_response_english = "Hi there!"

    bot_response = translate_text(translator_client, bot_response_english, current_lang, DEFAULT_LANGUAGE)
    st.session_state.chat_history.append((user_input_original, bot_response))

    try:
        from forwarding import save_user_interaction
        session_id = getattr(st.session_state, 'session_id', None)
        if classification_category != 'Casual Greeting' and not bot_response_english.startswith("Sorry"):
            save_user_interaction(user_input_english, bot_response_english, session_id)
    except Exception as e:
        st.error(f"Error saving interaction: {e}")

    st.session_state.user_input = ""

def display_chat():
    import os
    os.makedirs("tts_audio", exist_ok=True)

    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        message(user_msg, is_user=True, key=f"user_msg_{i}")
        message(bot_msg, key=f"bot_msg_{i}")

        text_hash = hashlib.md5(bot_msg.encode('utf-8')).hexdigest()
        audio_path = f"tts_audio/{text_hash}_{st.session_state.selected_language}.mp3"

        if bot_msg and texttospeech_client:
            lang_code = st.session_state.selected_language
            try:
                synthesis_input = texttospeech.SynthesisInput(text=bot_msg)
                voice_name_map = {
                    'en': 'en-US-Wavenet-C',
                    'hi': 'hi-IN-Wavenet-C',
                    'ta': 'ta-IN-Wavenet-C',
                    'te': 'te-IN-Standard-A'
                }
                voice_params = texttospeech.VoiceSelectionParams(
                    language_code=lang_code,
                    name=voice_name_map.get(lang_code),
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
                )
                audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
                response_tts = texttospeech_client.synthesize_speech(
                    request={"input": synthesis_input, "voice": voice_params, "audio_config": audio_config}
                )
                with open(audio_path, "wb") as out:
                    out.write(response_tts.audio_content)
            except Exception as e:
                st.warning(f"TTS generation failed: {e}")
                audio_path = None

        if audio_path and os.path.exists(audio_path):
            with st.container():
                if st.button(f"🔊 Play Response {i}", key=f"play_audio_btn_{i}"):
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")

def set_language():
    st.session_state.selected_language = st.session_state.lang_selector
    st.session_state.chat_history = []
    st.rerun()

def main():
    st.set_page_config(page_title="Multilingual Animal Bites Chatbot", layout="centered")
    st.title("Chatbot for Animal Bites")

    lang_codes = list(SUPPORTED_LANGUAGES.keys())
    if st.session_state.selected_language not in lang_codes:
        st.session_state.selected_language = DEFAULT_LANGUAGE

    try:
        current_index = lang_codes.index(st.session_state.selected_language)
    except ValueError:
        current_index = 0

    st.sidebar.selectbox(
        "Select Language",
        options=lang_codes,
        format_func=lambda code: SUPPORTED_LANGUAGES.get(code, code),
        key="lang_selector",
        on_change=set_language,
        index=current_index
    )

    display_chat()

    placeholder_text = translate_text(translator_client, "Enter your message here", st.session_state.selected_language, DEFAULT_LANGUAGE)
    st.text_input(
        "Type something...",
        key="user_input",
        placeholder=placeholder_text or "Enter your message here",
        on_change=process_input
    )

if __name__ == "__main__":
    main()
