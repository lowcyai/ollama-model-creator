import argparse
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

load_dotenv()


def parse_hf_url(url_or_repo: str) -> tuple[str, Optional[str]]:
    url_or_repo = url_or_repo.strip()

    if url_or_repo.startswith(("http://", "https://")):
        parsed = urlparse(url_or_repo)
        path_parts = [p for p in parsed.path.split("/") if p]

        if len(path_parts) >= 2:
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            filename = None
            if len(path_parts) >= 5 and path_parts[2] in ("blob", "resolve"):
                filename = "/".join(path_parts[4:])
            return repo_id, filename

    parts = url_or_repo.split("/")
    if len(parts) == 2:
        return url_or_repo, None
    elif len(parts) > 2:
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])
        return repo_id, filename

    return url_or_repo, None


def build_modelfile_content(
    from_path: str,
    params: Optional[dict] = None,
    template: Optional[str] = None,
    stop_tokens: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    lines = [f"FROM {from_path}", ""]

    if params:
        for key, val in params.items():
            if val is not None and str(val).strip():
                lines.append(f"PARAMETER {key} {val}")
        lines.append("")

    if stop_tokens:
        for token in stop_tokens:
            if token.strip():
                lines.append(f'PARAMETER stop "{token.strip()}"')
        lines.append("")

    if template and template.strip():
        lines.append(f'TEMPLATE """{template.strip()}"""')
        lines.append("")

    if system_prompt and system_prompt.strip():
        lines.append(f'SYSTEM """\n{system_prompt.strip()}\n"""')
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def create_starter_modelfile(
    repo_folder_name: str,
    filename: Optional[str] = None,
    modelfiles_dir: str | Path = "./modelfiles",
    models_dir_prefix: str = "../models",
) -> Path:
    target_dir = Path(modelfiles_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    modelfile_path = target_dir / f"{repo_folder_name.lower()}.modelfile"
    from_path = f"{models_dir_prefix}/{repo_folder_name}/{filename}" if filename else f"{models_dir_prefix}/{repo_folder_name}"

    if not modelfile_path.exists():
        default_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_ctx": 8192,
        }
        default_system = "Jesteś pomocnym asystentem AI. Odpowiadasz precyzyjnie w języku polskim."
        content = build_modelfile_content(
            from_path=from_path,
            params=default_params,
            system_prompt=default_system,
        )
        modelfile_path.write_text(content, encoding="utf-8")

    return modelfile_path


def download_hf_model(
    model_url_or_repo: str,
    filename: Optional[str] = None,
    output_dir: str | Path = "./models",
    token: Optional[str] = None,
) -> tuple[str, Path, str, Optional[str]]:
    hf_token = token or os.getenv("HF_TOKEN")
    if hf_token == "hf_your_token_here":
        hf_token = None

    repo_id, extracted_filename = parse_hf_url(model_url_or_repo)
    final_filename = filename or extracted_filename

    repo_folder_name = repo_id.replace("/", "_")
    target_dir = Path(output_dir) / repo_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if final_filename:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=final_filename,
            local_dir=str(target_dir),
            token=hf_token,
        )
        return repo_id, Path(file_path), repo_folder_name, final_filename

    download_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        token=hf_token,
    )
    return repo_id, Path(download_dir), repo_folder_name, None


def main():
    parser = argparse.ArgumentParser(
        description="Pobieranie modeli z HuggingFace i generowanie pliku Modelfile dla Ollama."
    )
    parser.add_argument("model", help="URL z HuggingFace lub nazwa repozytorium (np. unsloth/Ministral-3-8B-Instruct-2512-GGUF)")
    parser.add_argument("-f", "--filename", default=None, help="Nazwa konkretnego pliku do pobrania (np. model.gguf)")
    parser.add_argument("-o", "--output-dir", default="./models", help="Katalog docelowy na pobrane pliki modeli")
    parser.add_argument("-t", "--token", default=None, help="Token API HuggingFace")

    args = parser.parse_args()

    try:
        repo_id, path_or_file, repo_folder_name, final_filename = download_hf_model(
            model_url_or_repo=args.model,
            filename=args.filename,
            output_dir=args.output_dir,
            token=args.token,
        )
        print(f"Pobrano model: {repo_id}")
        print(f"Lokalizacja: {path_or_file}")

        modelfile_path = create_starter_modelfile(repo_folder_name, final_filename)
        print(f"Utworzono Modelfile: {modelfile_path}")

        model_alias = repo_folder_name.lower().replace("_", "-")
        print("\nAby zarejestrować model w Ollama, uruchom:")
        print(f"  ollama create {model_alias} -f {modelfile_path}")
        print(f"  ollama run {model_alias}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()
