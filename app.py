import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import whisper
import google.generativeai as genai
from fpdf import FPDF
import os
import uuid
from datetime import datetime
from typing import Optional


# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Transcrição de Aula com Gemini", layout="wide")

# --- Configuration ---

# 1. API Key (Using Environment Variable)
gemini_api_key = 'AIzaSyCk1DipP7yvFE52g8R9c2mVRX9mIv1kRtY' # Read from environment variable

if not gemini_api_key:
    st.error("❗️ Erro: Variável de ambiente 'GEMINI_API_KEY' não encontrada.")
    st.info(
        "Por favor, defina a variável de ambiente GEMINI_API_KEY com sua chave API Gemini.\n"
        "Exemplo (Linux/macOS): export GEMINI_API_KEY='SUA_CHAVE_AQUI'\n"
        "Exemplo (Windows CMD): set GEMINI_API_KEY=SUA_CHAVE_AQUI\n"
        "Exemplo (Windows PowerShell): $env:GEMINI_API_KEY='SUA_CHAVE_AQUI'"
    )
    st.stop() # Stop execution if key is missing
else:
    try:
        genai.configure(api_key=gemini_api_key)
        # --- THE FIX IS HERE ---
        gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest") # Use the versioned model name
        # -----------------------
        # Optional: Indicate success, maybe later in the sidebar if needed
    except Exception as e:
        st.error(f"🚨 Erro ao configurar a API Gemini com a chave da variável de ambiente: {e}")
        st.stop()


# 2. History Directory
HISTORICO_DIR = "pdfs_gerados"
os.makedirs(HISTORICO_DIR, exist_ok=True)

# 3. Audio Settings
SAMPLERATE = 16000
CHANNELS = 1
AUDIO_QUEUE = queue.Queue()

# --- Model Loading (Cached) ---
@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model once and caches it."""
    st.info("Carregando modelo Whisper (pode levar um momento na primeira vez)...")
    try:
        model = whisper.load_model("base") # Or choose another model size like "small", "medium"
        st.success("Modelo Whisper carregado.")
        return model
    except Exception as e:
        st.error(f"🚨 Erro ao carregar o modelo Whisper: {e}")
        st.info("Verifique sua instalação do Whisper e dependências (como ffmpeg).")
        st.stop()

model_whisper = load_whisper_model()

# --- Core Functions ---

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(f"Warning in audio callback: {status}") # Log to console
    AUDIO_QUEUE.put(indata.copy())

def gravar_audio_streamlit(duration: int) -> Optional[np.ndarray]:
    """
    Grava áudio do microfone pela duração especificada usando sounddevice.
    """
    st.info(f"🎙️ Gravando áudio por {duration} segundos...")
    while not AUDIO_QUEUE.empty():
        try:
            AUDIO_QUEUE.get_nowait()
        except queue.Empty:
            break

    audio_data = []
    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, callback=audio_callback):
            progress_bar = st.progress(0)
            for i in range(duration):
                sd.sleep(1000)
                progress_bar.progress((i + 1) / duration)
            sd.sleep(200)

            while not AUDIO_QUEUE.empty():
                try:
                    data = AUDIO_QUEUE.get_nowait()
                    audio_data.append(data)
                except queue.Empty:
                    break

        if not audio_data:
            st.warning("⚠️ Nenhuma data de áudio capturada. Verifique seu microfone.")
            return None

        audio_np = np.concatenate(audio_data, axis=0)
        st.success("✅ Gravação finalizada.")
        return audio_np.flatten()

    except sd.PortAudioError as pae:
        st.error(f"🚨 Erro de Áudio (PortAudio): {pae}")
        st.error("Verifique se o microfone está conectado e configurado corretamente.")
        return None
    except Exception as e:
        st.error(f"🚨 Erro durante a gravação: {e}")
        return None


def transcrever_audio(audio: np.ndarray) -> Optional[str]:
    """
    Transcreve o áudio usando o modelo Whisper.
    """
    if model_whisper is None:
         st.error("🚨 Modelo Whisper não está carregado.")
         return None

    st.info("🔄 Transcrevendo áudio com Whisper...")
    try:
        if audio.dtype != np.float32:
             audio = audio.astype(np.float32)
        result = model_whisper.transcribe(audio, fp16=False)
        st.success("✅ Transcrição concluída.")
        return result["text"]
    except Exception as e:
        st.error(f"🚨 Erro durante a transcrição: {e}")
        return None


def organizar_texto_gemini(texto_transcrito: str) -> Optional[str]:
    """
    Organiza o texto transcrito usando a API Gemini.
    """
    # Check if gemini_model was successfully initialized earlier
    if 'gemini_model' not in globals() or gemini_model is None:
        st.error("🚨 Modelo Gemini não inicializado corretamente.")
        return None

    st.info("✨ Organizando texto com Gemini...")
    prompt = f"""
    **Tarefa:** Organize o seguinte texto(Formatando em markup), que é uma transcrição de uma aula ou reunião,
    em um formato claro e legível. Use títulos, subtítulos, marcadores (bullets ou listas numeradas)
    e quebras de linha para estruturar o conteúdo de forma lógica. O objetivo é criar um
    resumo bem organizado dos pontos principais discutidos.

    **Texto Transcrito:**
    ```
    {texto_transcrito}
    ```

    **Texto Organizado:**
    """
    try:
        # Now using the correctly initialized gemini_model
        resposta = gemini_model.generate_content(prompt)

        if not resposta.candidates:
            st.warning("⚠️ Gemini não retornou conteúdo.")
            try:
                 block_reason = resposta.prompt_feedback.block_reason
                 st.warning(f"   Motivo do bloqueio: {block_reason}")
            except Exception:
                 st.warning("   Não foi possível obter o motivo do bloqueio.")
            return None

        texto_organizado = resposta.text
        st.success("✅ Texto organizado pelo Gemini.")
        return texto_organizado

    except Exception as e:
        st.error(f"🚨 Erro ao chamar a API Gemini: {e}")
        if hasattr(e, 'message'):
            st.error(f"   Detalhe: {e.message}")
        return None


def gerar_pdf(texto_organizado: str) -> Optional[str]:
    """
    Gera um arquivo PDF a partir do texto organizado.
    """
    st.info("📄 Gerando PDF...")
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:6]
        nome_arquivo = f"aula_{timestamp}_{unique_id}.pdf"
        caminho_arquivo = os.path.join(HISTORICO_DIR, nome_arquivo)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Processar o texto para ajustar a formatação
        for linha in texto_organizado.split('\n'):
            try:
                # Detecta título com "<b>"
                if '<b>' in linha:
                    pdf.set_font("Arial", 'B', 16)  # Aplica negrito
                    linha = linha.replace("<b>", "").replace("</b>", "")  # Remove as tags <b> e </b>
                # Detecta itálico com "<i>"
                elif '<i>' in linha:
                    pdf.set_font("Arial", 'I', 12)  # Aplica itálico
                    linha = linha.replace("<i>", "").replace("</i>", "")  # Remove as tags <i> e </i>
                else:
                    pdf.set_font("Arial", '', 12)  # Texto normal

                # Adiciona linha no PDF
                linha_encoded = linha.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, linha_encoded)

            except Exception as pdf_err:
                print(f"PDF encoding/writing error for line: {linha}. Error: {pdf_err}")
                pdf.multi_cell(0, 10, "[linha com caracteres não suportados]")

        pdf.output(caminho_arquivo)
        st.success(f"✅ PDF gerado: {nome_arquivo}")
        return caminho_arquivo
    except Exception as e:
        st.error(f"🚨 Erro ao gerar o PDF: {e}")
        return None





def listar_pdfs() -> list[str]:
    """Lista os arquivos PDF no diretório de histórico."""
    try:
        arquivos = [f for f in os.listdir(HISTORICO_DIR) if f.endswith(".pdf")]
        arquivos.sort(key=lambda f: os.path.getmtime(os.path.join(HISTORICO_DIR, f)), reverse=True)
        return arquivos
    except Exception as e:
        st.error(f"🚨 Erro ao listar PDFs: {e}")
        return []

# --- Streamlit Interface ---

st.title("📚 Transcrição Inteligente de Aulas e Reuniões")
st.markdown("Grave áudio, transcreva com Whisper e organize com Gemini AI.")

# Sidebar Menu
aba = st.sidebar.radio("Menu", ["🎙️ Nova Transcrição", "📁 Histórico de PDFs"], key="menu_principal")
st.sidebar.markdown("---")
if gemini_api_key:
     st.sidebar.success("API Gemini Conectada")
st.sidebar.info("Desenvolvido com Streamlit, Whisper & Gemini.")


# --- Aba: Nova Transcrição ---
if aba == "🎙️ Nova Transcrição":
    st.header("🎤 Grave, Transcreva e Organize")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controles de Gravação")
        duracao = st.slider(
            "Duração da gravação (segundos): maximo permitido 1h20min",
            min_value=10,
            max_value=4800,
            value=60,
            step=3,
            key="duracao_slider",
            help="Defina por quantos segundos o áudio será gravado."
        )

        start_button_pressed = st.button("▶️ Iniciar Gravação", key="start_button")

        if start_button_pressed:
            st.session_state.pop('audio_gravado', None)
            st.session_state.pop('texto_transcrito', None)
            st.session_state.pop('texto_organizado', None)
            st.session_state.pop('caminho_pdf', None)

            audio_gravado = gravar_audio_streamlit(duracao)

            if audio_gravado is not None:
                st.session_state.audio_gravado = audio_gravado

                with st.spinner("⏳ Processando áudio... (Transcrição e Organização)"):
                    texto_transcrito = transcrever_audio(st.session_state.audio_gravado)
                    st.session_state.texto_transcrito = texto_transcrito

                    if texto_transcrito:
                        # Check again if gemini_model is available before calling
                        if 'gemini_model' in globals() and gemini_model is not None:
                            texto_organizado = organizar_texto_gemini(texto_transcrito)
                            st.session_state.texto_organizado = texto_organizado

                            if texto_organizado:
                                caminho_pdf = gerar_pdf(texto_organizado)
                                st.session_state.caminho_pdf = caminho_pdf
                        else:
                            # This case should ideally not happen if startup checks pass
                            st.error("Erro crítico: Modelo Gemini não está disponível para organização.")


            if st.session_state.get('audio_gravado') is not None and not st.session_state.get('caminho_pdf'):
                 st.warning("⚠️ Processamento não concluído totalmente. Verifique mensagens de erro acima.")


    with col2:
        st.subheader("Resultados")

        texto_transcrito_atual = st.session_state.get('texto_transcrito')
        texto_organizado_atual = st.session_state.get('texto_organizado')
        caminho_pdf_atual = st.session_state.get('caminho_pdf')

        if texto_transcrito_atual:
             with st.expander("Ver Texto Transcrito Bruto (Whisper)"):
                  st.text_area("Transcrição", texto_transcrito_atual, height=150, key="transcricao_bruta_area", disabled=True)

        if texto_organizado_atual:
            st.text_area("📝 Texto Organizado (Gemini)", texto_organizado_atual, height=400, key="texto_organizado_area")

            if caminho_pdf_atual:
                try:
                    if os.path.exists(caminho_pdf_atual):
                        with open(caminho_pdf_atual, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button(
                            label="📥 Baixar PDF Organizado",
                            data=pdf_bytes,
                            file_name=os.path.basename(caminho_pdf_atual),
                            mime="application/pdf",
                            key="download_pdf_button"
                        )
                    else:
                        st.error(f"🚨 Erro: Arquivo PDF '{os.path.basename(caminho_pdf_atual)}' não encontrado no caminho esperado.")

                except Exception as e:
                    st.error(f"🚨 Erro ao preparar PDF para download: {e}")
        elif start_button_pressed:
             st.info("Aguardando resultados da gravação e processamento...")
        else:
             st.info("Clique em 'Iniciar Gravação' para começar.")


# --- Aba: Histórico ---
elif aba == "📁 Histórico de PDFs":
    st.header("📚 Histórico de Transcrições Salvas")
    st.markdown("Faça o download dos PDFs gerados anteriormente.")

    arquivos_pdf = listar_pdfs()

    if not arquivos_pdf:
        st.info("ℹ️ Nenhum PDF foi gerado ainda.")
    else:
        st.write(f"Total de arquivos: {len(arquivos_pdf)}")
        num_cols = 3
        cols = st.columns(num_cols)
        for i, arquivo in enumerate(arquivos_pdf):
            col_index = i % num_cols
            with cols[col_index]:
                caminho_completo = os.path.join(HISTORICO_DIR, arquivo)
                try:
                    if os.path.exists(caminho_completo):
                        with open(caminho_completo, "rb") as f:
                            pdf_bytes_hist = f.read()
                        st.download_button(
                            label=f"📄 {arquivo}",
                            data=pdf_bytes_hist,
                            file_name=arquivo,
                            mime="application/pdf",
                            key=f"hist_{arquivo}"
                        )
                    else:
                        st.warning(f"Arquivo não encontrado: {arquivo}")

                except Exception as e:
                     st.error(f"Erro ao ler {arquivo}: {e}")

# --- Footer ---
st.markdown("---")