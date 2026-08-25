# Ollama Model Creator – Kreator i Menedżer Modeli dla Ollama

Narzędzie ułatwiające pobieranie wag modeli z Hugging Face (pojedyncze pliki `.gguf` lub całe repozytoria), tworzenie i dostrajanie konfiguracji `Modelfile` oraz automatyczną rejestrację modeli w lokalnej instancji [Ollama](https://ollama.com).

Projekt oferuje zarówno nowoczesny interfejs graficzny w **Streamlit**, jak i szybki interfejs wiersza poleceń (**CLI**).

---

## 🚀 Funkcjonalności

- **Pobieranie z Hugging Face**:
  - Pobieranie pojedynczych plików kwantyzacji (np. `.gguf`) lub pełnych migawek repozytoriów.
  - Obsługa tokenów API (`HF_TOKEN`) dla modeli wymagających autoryzacji (gated / private).
  - Automatyczne parsowanie linków i identyfikatorów repozytoriów z Hugging Face.
- **Wizualny Konfigurator Modelfile**:
  - Wybór pobranych modeli jako bazy (`FROM`).
  - Regulacja kluczowych parametrów: okno kontekstu (`num_ctx`), `temperature`, `top_p`, `min_p`, `repeat_penalty` oraz liczba wątków CPU (`num_thread`).
  - Gotowe szablony konwersacyjne: **Mistral / Ministral ([INST])**, **ChatML (<|im_start|>)**, **Llama 3 (<|start_header_id|>)** oraz możliwość definiowania własnego szablonu.
  - Definiowanie promptu systemowego (`SYSTEM`) oraz tokenów zatrzymania (`PARAMETER stop`).
  - Podgląd kodu `Modelfile` na żywo z funkcją zapisu do pliku.
- **Bezpośrednia Integracja z Ollama**:
  - Rejestracja utworzonego modelu jednym kliknięciem (`ollama create`).
  - Podgląd zainstalowanych modeli (`ollama list`) bez opuszczania aplikacji.
- **Dwa tryby pracy**:
  - Pełny interfejs graficzny (Web UI w Streamlit).
  - Skrypt wiersza poleceń (`download_model.py`) do automatyzacji i skryptów bash.

---

## 📁 Struktura Projektu

```text
ollama-model-creator/
├── app.py               # Główna aplikacja webowa w Streamlit
├── download_model.py    # Moduł pobierania modeli HF i skrypt CLI
├── requirements.txt     # Zależności projektu w Pythonie
├── .env.example         # Przykładowy plik konfiguracyjny zmiennych środowiskowych
├── .gitignore           # Konfiguracja ignorowania wag modeli i plików tymczasowych
├── models/              # Katalog docelowy na pobrane pliki i foldery modeli
└── modelfiles/          # Katalog na wygenerowane pliki .modelfile
```

---

## 📋 Wymagania

- **Python 3.10+**
- **Ollama** zainstalowana i uruchomiona w systemie ([pobierz Ollama](https://ollama.com/download))
- (Opcjonalnie) Konto i token API na **Hugging Face** (dla modeli wymagających zgody licencyjnej, np. Llama lub Gemma)

---

## 🛠️ Instalacja

1. **Sklonuj repozytorium:**

```bash
git clone https://github.com/lowcyai/ollama-model-creator.git
cd ollama-model-creator
```

2. **Utwórz i aktywuj środowisko wirtualne:**

```bash
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

3. **Zainstaluj wymagane pakiety:**

```bash
pip install -r requirements.txt
```

4. **(Opcjonalnie) Skonfiguruj token Hugging Face:**

Skopiuj plik `.env.example` do `.env` i wpisz swój token:

```bash
cp .env.example .env
```

Edytuj plik `.env`:

```env
HF_TOKEN=hf_twoj_token_z_huggingface
```

---

## 🖥️ Użycie

### 1. Interfejs Graficzny (Streamlit)

Uruchom aplikację webową poleceniem:

```bash
streamlit run app.py
```

Aplikacja otworzy się w przeglądarce (domyślnie pod adresem `http://localhost:8501`). Składa się z 3 intuicyjnych zakładek:

1. **📥 1. Pobierz Model**:
   - Wklej adres URL lub nazwę repozytorium z Hugging Face (np. `unsloth/Ministral-3-8B-Instruct-2512-GGUF`).
   - Opcjonalnie podaj nazwę konkretnego pliku `.gguf` (np. `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf`).
   - Kliknij **Pobierz Model**.
2. **⚙️ 2. Konfigurator Modelfile**:
   - Wybierz pobrany plik z listy.
   - Ustaw parametry generowania, wybierz szablon promptu oraz zdefiniuj instrukcję systemową.
   - Kliknij **Zapisz Modelfile**.
3. **🚀 3. Rejestracja w Ollama**:
   - Wybierz zapisany plik `.modelfile`.
   - Nadaj modelowi nazwę (alias).
   - Kliknij **Utwórz model w Ollama** – model zostanie skompilowany i dodany do Twojej lokalnej bazy Ollama.

---

### 2. Wiersz poleceń (CLI)

Możesz także pobrać model i wygenerować startowy `Modelfile` bezpośrednio z terminala:

```bash
# Pobranie konkretnego pliku GGUF
python download_model.py unsloth/Ministral-3-8B-Instruct-2512-GGUF -f Ministral-3-8B-Instruct-2512-Q4_K_M.gguf

# Pobranie z podaniem tokenu API
python download_model.py meta-llama/Llama-3.2-3B-Instruct-GGUF -f Llama-3.2-3B-Instruct-Q4_K_M.gguf -t hf_twoj_token
```

Po zakończeniu pobierania skrypt automatycznie utworzy szablon pliku w folderze `modelfiles/` i wyświetli instrukcję rejestracji:

```bash
ollama create moj-model -f modelfiles/nazwa.modelfile
ollama run moj-model
```

---

## ⚙️ Kluczowe Parametry Modelfile

Aplikacja pozwala precyzyjnie kontrolować dyrektywy pliku konfiguracyjnego Ollama:

| Dyrektywa / Parametr | Opis |
| :--- | :--- |
| `FROM` | Ścieżka do bazowego pliku binarnego (np. pliku `.gguf`). |
| `PARAMETER num_ctx` | Długość kontekstu w tokenach (np. 8192, 32768, 131072). |
| `PARAMETER temperature` | Stopień losowości odpowiedzi (niższa = bardziej precyzyjna, wyższa = bardziej kreatywna). |
| `PARAMETER top_p` | Nucleus sampling – ograniczenie losowania do najbardziej prawdopodobnych tokenów. |
| `PARAMETER min_p` | Minimalny próg prawdopodobieństwa tokena w stosunku do najsilniejszego kandydata. |
| `PARAMETER repeat_penalty` | Kara za powtarzanie tych samych słów i fraz. |
| `PARAMETER num_thread` | Liczba wątków procesora przydzielona do obliczeń. |
| `PARAMETER stop` | Tokeny zatrzymujące generowanie (np. `<|im_end|>`, `[/INST]`, `<|eot_id|>`). |
| `TEMPLATE` | Struktura szablonu konwersacji dopasowana do architektury modelu. |
| `SYSTEM` | Główna instrukcja określająca rolę, styl i zasady odpowiedzi asystenta. |

---

## 🌐 Projekt dla [lowcyai.pl](https://lowcyai.pl)

Narzędzie stanowi część serii artykułów i wdrożeń publikowanych na blogu [lowcyai.pl](https://lowcyai.pl), poświęconym praktycznym zastosowaniom sztucznej inteligencji, lokalnym modelom LLM oraz automatyzacjom.

Więcej poradników, przykładów i automatyzacji znajdziesz na [lowcyai.pl](https://lowcyai.pl).

---

## 📬 Kontakt

Masz pytania, uwagi lub pomysł na rozwój narzędzia?
- Blog: [lowcyai.pl](https://lowcyai.pl)
- E-mail: [webmaster@lowcyai.pl](mailto:webmaster@lowcyai.pl)

---

## 📄 Licencja

Projekt udostępniany jest na licencji [MIT](LICENSE).

