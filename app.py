import os
import subprocess
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from download_model import build_modelfile_content, download_hf_model

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
MODELFILES_DIR = BASE_DIR / "modelfiles"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODELFILES_DIR.mkdir(parents=True, exist_ok=True)

PRESET_TEMPLATES = {
    "Mistral / Ministral ([INST])": """[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }} [/INST]""",
    "ChatML (<|im_start|>)": """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
""",
    "Llama 3 (<|start_header_id|>)": """<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    "Domyślny (z pliku GGUF)": "",
    "Własny": "",
}

PRESET_STOPS = {
    "Mistral / Ministral ([INST])": ["[INST]", "[/INST]"],
    "ChatML (<|im_start|>)": ["<|im_start|>", "<|im_end|>"],
    "Llama 3 (<|start_header_id|>)": ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],
    "Domyślny (z pliku GGUF)": [],
    "Własny": [],
}


def scan_downloaded_models() -> list[str]:
    items = []
    if not MODELS_DIR.exists():
        return items

    for root, _, files in os.walk(MODELS_DIR):
        rel_root = Path(root).relative_to(MODELS_DIR)
        for f in files:
            if f.endswith((".gguf", ".bin", ".safetensors")):
                rel_path = f"../models/{rel_root / f}" if str(rel_root) != "." else f"../models/{f}"
                items.append(rel_path)

        if str(rel_root) != "." and not any(f.endswith(".gguf") for f in files):
            items.append(f"../models/{rel_root}")

    return sorted(list(set(items)))


def scan_modelfiles() -> list[str]:
    if not MODELFILES_DIR.exists():
        return []
    return sorted([f.name for f in MODELFILES_DIR.glob("*.modelfile")])


st.set_page_config(
    page_title="Ollama Modelfile Manager",
    page_icon="🦙",
    layout="wide",
)

st.title("🦙 Ollama Modelfile Manager")
st.caption("Narzędzie do pobierania modeli z HuggingFace, konfiguracji Modelfile oraz rejestracji w Ollama")

env_token = os.getenv("HF_TOKEN")

with st.sidebar:
    st.header("Informacje")
    if env_token and env_token != "hf_your_token_here":
        st.success("Wykryto HF_TOKEN w środowisku")
    else:
        st.info("Brak HF_TOKEN (wymagany dla modeli z ograniczonym dostępem)")

    st.divider()
    downloaded_models = scan_downloaded_models()
    existing_modelfiles = scan_modelfiles()
    st.metric("Pobrane modele", len(downloaded_models))
    st.metric("Zapisane Modelfiles", len(existing_modelfiles))

tab_download, tab_config, tab_ollama = st.tabs([
    "📥 1. Pobierz Model",
    "⚙️ 2. Konfigurator Modelfile",
    "🚀 3. Rejestracja w Ollama",
])

with tab_download:
    st.header("Pobieranie modeli z HuggingFace")

    col1, col2 = st.columns([3, 2])
    with col1:
        model_url_input = st.text_input(
            "Adres URL lub ID repozytorium HuggingFace:",
            placeholder="np. unsloth/Ministral-3-8B-Instruct-2512-GGUF",
        )
        filename_input = st.text_input(
            "Nazwa konkretnego pliku (opcjonalnie):",
            placeholder="np. Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
        )
    with col2:
        hf_token_input = st.text_input(
            "Token HuggingFace API (opcjonalnie):",
            value=env_token if env_token and env_token != "hf_your_token_here" else "",
            type="password",
        )

    if st.button("Pobierz Model", type="primary", use_container_width=True):
        if not model_url_input.strip():
            st.error("Podaj adres URL lub ID repozytorium.")
        else:
            with st.spinner("Pobieranie modelu z HuggingFace..."):
                try:
                    repo_id, path_or_file, repo_folder_name, final_filename = download_hf_model(
                        model_url_or_repo=model_url_input,
                        filename=filename_input.strip() if filename_input else None,
                        output_dir=str(MODELS_DIR),
                        token=hf_token_input.strip() if hf_token_input else None,
                    )
                    st.success(f"Pobrano pomyślnie: `{path_or_file}`")
                except Exception as e:
                    st.error(f"Błąd podczas pobierania: {e}")

with tab_config:
    st.header("Konfiguracja i Optymalizacja Modelfile")
    models_available = scan_downloaded_models()

    if not models_available:
        st.warning("Brak pobranych modeli w katalogu `models/`. Pobierz model w zakładce 1.")
        st.stop()

    col_model, col_modelfile_name = st.columns([3, 2])
    with col_model:
        selected_from_path = st.selectbox(
            "Wybierz model bazowy (FROM):",
            options=models_available,
        )

    with col_modelfile_name:
        if selected_from_path:
            clean_stem = Path(selected_from_path).stem.lower().replace("_", "-")
            default_file_name = f"{clean_stem}.modelfile"
        else:
            default_file_name = "custom.modelfile"
        target_modelfile_name = st.text_input("Nazwa pliku Modelfile:", value=default_file_name)

    st.subheader("Parametry")
    col_ctx, col_temp, col_topp = st.columns(3)
    with col_ctx:
        num_ctx = st.select_slider(
            "Rozmiar okna kontekstu (num_ctx):",
            options=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
            value=8192,
        )
    with col_temp:
        temperature = st.slider("Temperatura (temperature):", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    with col_topp:
        top_p = st.slider("Top P (top_p):", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

    col_minp, col_rep, col_thread = st.columns(3)
    with col_minp:
        min_p = st.slider("Min P (min_p):", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    with col_rep:
        repeat_penalty = st.slider("Kara za powtórzenia (repeat_penalty):", min_value=1.0, max_value=2.0, value=1.1, step=0.05)
    with col_thread:
        num_thread = st.number_input("Wątki CPU (num_thread, 0 = auto):", min_value=0, max_value=64, value=0)

    st.subheader("Szablon i Instrukcja")
    preset_choice = st.selectbox("Szablon wiadomości:", options=list(PRESET_TEMPLATES.keys()))
    template_input = st.text_area("Szablon (TEMPLATE):", value=PRESET_TEMPLATES[preset_choice], height=120)

    stops_input_str = st.text_input(
        "Tokeny zatrzymania (stop, oddzielone przecinkami):",
        value=", ".join(PRESET_STOPS[preset_choice]),
    )
    stop_tokens_list = [s.strip() for s in stops_input_str.split(",") if s.strip()]

    system_prompt = st.text_area(
        "Instrukcja systemowa (SYSTEM):",
        value="Jesteś pomocnym i precyzyjnym asystentem AI. Odpowiadasz w języku polskim.",
        height=90,
    )

    params_dict = {
        "num_ctx": num_ctx,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
    }
    if num_thread > 0:
        params_dict["num_thread"] = num_thread

    generated_content = build_modelfile_content(
        from_path=selected_from_path,
        params=params_dict,
        template=template_input if template_input.strip() else None,
        stop_tokens=stop_tokens_list if stop_tokens_list else None,
        system_prompt=system_prompt if system_prompt.strip() else None,
    )

    st.subheader("Podgląd Modelfile")
    st.code(generated_content, language="dockerfile")

    if st.button("Zapisz Modelfile", type="primary", use_container_width=True):
        if not target_modelfile_name.strip():
            st.error("Podaj poprawną nazwę pliku.")
        else:
            save_path = MODELFILES_DIR / target_modelfile_name.strip()
            save_path.write_text(generated_content, encoding="utf-8")
            st.success(f"Zapisano: `{save_path}`")

with tab_ollama:
    st.header("Rejestracja w Ollama")
    modelfiles_list = scan_modelfiles()

    if not modelfiles_list:
        st.warning("Brak plików `.modelfile` w katalogu `modelfiles/`.")
        st.stop()

    col_mf, col_alias = st.columns([3, 2])
    with col_mf:
        selected_modelfile = st.selectbox("Wybierz plik Modelfile:", options=modelfiles_list)
    with col_alias:
        default_alias = selected_modelfile.replace(".modelfile", "").lower().replace("_", "-")
        model_alias = st.text_input("Nazwa modelu w Ollama:", value=default_alias)

    full_modelfile_path = f"modelfiles/{selected_modelfile}"
    st.code(f"ollama create {model_alias} -f {full_modelfile_path}", language="bash")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Utwórz model w Ollama (`ollama create`)", type="primary", use_container_width=True):
            if not model_alias.strip():
                st.error("Podaj nazwę modelu.")
            else:
                with st.spinner(f"Rejestracja modelu '{model_alias}' w Ollama..."):
                    try:
                        result = subprocess.run(
                            ["ollama", "create", model_alias.strip(), "-f", full_modelfile_path],
                            cwd=str(BASE_DIR),
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            st.success(f"Model **{model_alias}** został pomyślnie zarejestrowany!")
                            if result.stdout:
                                st.code(result.stdout, language="text")
                        else:
                            st.error("Błąd podczas tworzenia modelu:")
                            st.code(result.stderr, language="text")
                    except FileNotFoundError:
                        st.error("Komenda `ollama` nie została znaleziona w systemie.")
                    except Exception as e:
                        st.error(f"Wyjątek: {e}")

    with col_btn2:
        if st.button("Lista modeli (`ollama list`)", use_container_width=True):
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.code(result.stdout, language="text")
                else:
                    st.error("Błąd:")
                    st.code(result.stderr, language="text")
            except FileNotFoundError:
                st.error("Komenda `ollama` nie została znaleziona w systemie.")
            except Exception as e:
                st.error(f"Wyjątek: {e}")
