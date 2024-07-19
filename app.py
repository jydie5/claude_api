import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import base64
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
import anthropic
import re

# OpenAIクライアントの初期化
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Anthropicクライアントの初期化
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ページ初期設定
st.set_page_config(
    page_title="Integrated AI Chat Client",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設定ファイルの読み込み
def load_config():
    with open('.config.yaml') as file:
        return yaml.load(file, Loader=SafeLoader)

# 認証の初期化
def init_authenticator(config):
    return stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )

# 音声をテキストに変換する関数
def audio_to_text(audio_bytes):
    write_audio_file("recorded_audio.wav", audio_bytes)
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=open("recorded_audio.wav", mode="rb"),
    )
    return transcript.text

# 音声ファイルを書き込む関数
def write_audio_file(file_path, audio_bytes):
    with open(file_path, "wb") as audio_file:
        audio_file.write(audio_bytes)

# テキストを音声に変換する関数
def text_to_speech(text, voice):
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3"
    )
    audio_data = response.content
    
    # ローカルにMP3ファイルを保存
    with open("tts.mp3", "wb") as f:
        f.write(audio_data)
    
    return audio_data

def get_response(prompt, chat_history, output_container):
    response = ""
    
    if st.session_state.api_provider == "OpenAI":
        # OpenAI streaming
        openai_stream = openai_client.chat.completions.create(
            model=st.session_state.model,
            messages=[
                *[
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                    for message in chat_history
                ],
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
        )

        for chunk in openai_stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                output_container.markdown(response)
    
    elif st.session_state.api_provider == "Anthropic":
        # Anthropic streaming
        with anthropic_client.messages.stream(
            max_tokens=4096,
            messages=[
                *[
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                    for message in chat_history
                ],
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=st.session_state.model,
        ) as stream:
            for text in stream.text_stream:
                response += text
                output_container.markdown(response)
    
    return response

# トピックを抽出する関数
def extract_topic(user_input, assistant_response):
    user_words = set(user_input.lower().split())
    assistant_words = set(assistant_response.lower().split())
    common_words = user_words.intersection(assistant_words)
    if common_words:
        topic = max(common_words, key=len)
        return re.sub(r'[^a-zA-Z0-9]+', '_', topic)  # 特殊文字をアンダースコアに置換
    return None

# 会話履歴をMarkdownファイルに保存する関数
def save_conversation_history(conversation_history, topic=None):
    if topic:
        file_name = f"{topic}.md"
    else:
        file_name = "conversation.md"  # デフォルトのファイル名
    with open(file_name, "w") as file:
        for message in conversation_history:
            if message["role"] == "user":
                file.write(f"User: {message['content']}\n\n")
            else:
                file.write(f"Assistant: {message['content']}\n\n")
    return file_name

def main():
    # 設定ファイルの読み込み
    config = load_config()

    # 認証の初期化
    authenticator = init_authenticator(config)

    # サイドバーの画像表示
    image_path = "./static/ちいかわポップコーン_blurred.jpg"
    st.sidebar.image(image_path, use_column_width=True)

    # 認証用サイドバー
    with st.sidebar:
        authenticator.login()

    if st.session_state["authentication_status"]:
        st.sidebar.write(f'Login Name *{st.session_state["name"]}*')

        # メイン画面でアプリの実装
        st.title('Integrated AI Chat Client')

        # アプリ用サイドバー
        with st.sidebar:
            st.markdown("## おしゃべり機能")
            if "enable_tts" not in st.session_state:
                st.session_state.enable_tts = False  # デフォルトでTTSを無効化
            enable_tts = st.checkbox("Enable Text-to-Speech", st.session_state.enable_tts)
            st.session_state.enable_tts = enable_tts

            if "voice" not in st.session_state:
                st.session_state.voice = "nova"

            voice = st.selectbox("TTS Voice in OpenAI", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=4)
            st.session_state.voice = voice

            # ローカルのtts.mp3を再生するボタンを追加
            if os.path.exists("tts.mp3"):
                if st.button("もう一度おしゃべり"):
                    audio_file = open("tts.mp3", "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")

            # API提供元とモデル選択
            st.markdown("## API Provider and Model Selection")
            api_provider = st.selectbox("Choose API Provider", ["OpenAI", "Anthropic"])
            st.session_state.api_provider = api_provider

            if api_provider == "OpenAI":
                model_options = {
                    "gpt-4o-mini (Fast responses)": "gpt-4o-mini",
                    "gpt-4o (Advanced model)": "gpt-4o",
                }
            else:  # Anthropic
                model_options = {
                    "claude-3-haiku (Fast responses)": "claude-3-haiku-20240307",
                    #"claude-3-sonnet (General AI)": "claude-3-sonnet-20240229",
                    "claude-3-opus (OLD-Advanced model)": "claude-3-opus-20240229",
                    "claude-3-5-sonnet (Latest-Fast)": "claude-3-5-sonnet-20240620"
                }
            
            selected_model = st.selectbox(f"{api_provider} Model Type", list(model_options.keys()))
            st.session_state.model = model_options[selected_model]

            # 会話を保存するボタン
            st.markdown("## ログ保存")
            if st.button("会話履歴を保存"):
                topic = extract_topic(st.session_state.messages[-2]["content"], st.session_state.messages[-1]["content"])
                file_name = save_conversation_history(st.session_state.messages, topic)
                with open(file_name, "rb") as file:
                    st.download_button(
                        label="会話履歴をダウンロード",
                        data=file,
                        file_name=file_name,
                        mime="text/markdown"
                    )
            st.markdown("## ログアウト")  
                  
        authenticator.logout('Logout', 'sidebar')

        # セッション状態の初期化
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = []
            st.session_state.messages = []
            st.session_state.enable_tts = False  # デフォルトでTTSを無効化
        if "topics" not in st.session_state:
            st.session_state.topics = []

        # 新しいチャットボタン
        if st.button("New Chat"):
            st.session_state.chat_session = []
            st.session_state.messages = []
            st.session_state.topics = []

        # チャット履歴の表示
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.write("No conversation history.")

        # チャット入力
        input_text = st.chat_input("Type your message here...", key="input_text")

        # 音声入力
        audio_bytes = audio_recorder(
            text="<<<Recording ends,click again or remain silent for two seconds.>>>",
            recording_color="#e8b62c",
            neutral_color="#2EF218",
            icon_name="microphone-lines",
            icon_size="4x",
            pause_threshold=2.0,
            sample_rate=41_000
        )

        # 入力処理
        if input_text:
            st.session_state.messages.append({"role": "user", "content": input_text})
            with st.chat_message("user"):
                st.markdown(input_text)

            # 過去20回分の会話履歴を取得
            chat_history = st.session_state.chat_session[-20:]

            # ストリーミング出力用のコンテナを作成
            output_container = st.empty()

            full_response = get_response(input_text, chat_history, output_container)
            st.session_state.chat_session.append({"role": "user", "content": input_text})
            st.session_state.chat_session.append({"role": "assistant", "content": full_response})

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # トピックを抽出
            topic = extract_topic(input_text, full_response)
            if topic and topic not in st.session_state.topics:
                st.session_state.topics.append(topic)

            # テキストを音声に変換し、再生する (チェックボックスがオンの場合のみ)
            if st.session_state.enable_tts:
                audio_data = text_to_speech(full_response, st.session_state.voice)
                audio_str = f"data:audio/mpeg;base64,{base64.b64encode(audio_data).decode()}"
                audio_html = f"""
                                <audio autoplay>
                                <source src="{audio_str}" type="audio/mpeg">
                                Your browser does not support the audio element.
                                </audio>
                            """
                st.markdown(audio_html, unsafe_allow_html=True)

        elif audio_bytes:
            audio_transcript = audio_to_text(audio_bytes)
            if audio_transcript:
                st.session_state.messages.append({"role": "user", "content": audio_transcript})
                with st.chat_message("user"):
                    st.markdown(audio_transcript)

                # 過去20回分の会話履歴を取得
                chat_history = st.session_state.chat_session[-20:]

                # ストリーミング出力用のコンテナを作成
                output_container = st.empty()

                full_response = get_response(audio_transcript, chat_history, output_container)
                st.session_state.chat_session.append({"role": "user", "content": audio_transcript})
                st.session_state.chat_session.append({"role": "assistant", "content": full_response})

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # トピックを抽出
                topic = extract_topic(audio_transcript, full_response)
                if topic and topic not in st.session_state.topics:
                    st.session_state.topics.append(topic)

                # テキストを音声に変換し、再生する (チェックボックスがオンの場合のみ)
                if st.session_state.enable_tts:
                    audio_data = text_to_speech(full_response, st.session_state.voice)
                    audio_str = f"data:audio/mpeg;base64,{base64.b64encode(audio_data).decode()}"
                    audio_html = f"""
                                    <audio autoplay>
                                    <source src="{audio_str}" type="audio/mpeg">
                                    Your browser does not support the audio element.
                                    </audio>
                                """
                    st.markdown(audio_html, unsafe_allow_html=True)

    elif st.session_state["authentication_status"] is False:
        st.sidebar.error('Username/password is incorrect')

    elif st.session_state["authentication_status"] is None:
        st.sidebar.warning('Please enter your username and password')

if __name__ == "__main__":
    main()