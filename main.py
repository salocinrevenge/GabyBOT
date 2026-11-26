import base64
import gc
import html
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

import ollama
import torch
from ddgs import DDGS
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

MODEL = "gemma4:12b"
DIFFUSION_MODEL_ID = os.getenv(
    "DIFFUSION_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"
)
WORKSPACE = Path.cwd().resolve()
OUTPUT_DIR = WORKSPACE / "outputs"
UPLOADS_DIR = WORKSPACE / "uploads"
OFFLOAD_DIR = WORKSPACE / ".offload_cache"
MEMORY_FILE = WORKSPACE / "memoria.txt"
ALLOWLIST_FILE = WORKSPACE / ".agent_allowed_commands.json"
API_KEY_FILE = WORKSPACE / "ChaveAPI.txt"
CX_ID_FILE = WORKSPACE / "CX_ID.txt"

DEFAULT_ALLOWED_CMDS = ["ls", "pwd", "cat", "head", "tail", "grep", "find", "wc"]


def load_allowlist() -> set[str]:
    if ALLOWLIST_FILE.exists():
        try:
            return set(json.loads(ALLOWLIST_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return set(DEFAULT_ALLOWED_CMDS)


def save_allowlist(allowed: set[str]) -> None:
    ALLOWLIST_FILE.write_text(
        json.dumps(sorted(allowed), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


ALLOWED_COMMANDS = load_allowlist()


def _resolve_safe_path(rel_path: str) -> Path:
    """Garante que o agente opere apenas dentro da pasta do projeto."""
    target = (WORKSPACE / rel_path).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        raise PermissionError(f"Acesso negado fora do diretório do projeto: {rel_path}")
    return target


def _sanitize_filename(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower()).strip("_")
    return slug[:max_len] or "desenho"


def _load_search_credentials() -> tuple[str, str]:
    """Carrega API Key e CX ID dos arquivos locais ou variáveis de ambiente."""
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
    cx_id = os.getenv("GOOGLE_SEARCH_CX", "").strip()

    if not api_key and API_KEY_FILE.is_file():
        api_key = API_KEY_FILE.read_text(encoding="utf-8").strip()

    if not cx_id and CX_ID_FILE.is_file():
        cx_id = CX_ID_FILE.read_text(encoding="utf-8").strip()

    return api_key, cx_id


# [NOVO v3] Checagem real de RAM disponível no Linux para evitar travamento total (hard freeze)
def _get_available_ram_gb() -> float:
    """Retorna a quantidade de RAM disponível (MemAvailable) em GB."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 999.0


# [NOVO v3] Limpa imagens Base64 antigas do histórico para não inflar a RAM ao longo da conversa
def _prune_old_images_from_history(keep_last_n_msgs: int = 2) -> None:
    """Remove dados pesados de imagens (base64) das mensagens antigas do histórico."""
    cutoff = max(1, len(history) - keep_last_n_msgs)
    for i in range(1, cutoff):
        if isinstance(history[i], dict) and "images" in history[i]:
            del history[i]["images"]
            history[i]["content"] += " [Imagem removida da memória para economia de RAM]"


def _unload_llm(model_name: str) -> None:
    """Descarrega o LLM da VRAM/RAM antes de carregar o modelo de difusão."""
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
    except Exception:
        pass
    subprocess.run(
        ["ollama", "stop", model_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Aguarda o daemon do Ollama devolver a memória ao sistema operacional
    time.sleep(2.0)


def _has_thinking_loop(thought: str | None) -> bool:
    """Detecta loops pesados tanto por linhas repetidas quanto por blocos repetitivos."""
    if not thought:
        return False
    lines = [ln.strip() for ln in thought.splitlines() if ln.strip()]
    if len(lines) >= 8:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.35:
            return True
    if len(thought) > 1200:
        tail = thought[-400:]
        earlier = thought[:-400]
        if earlier.count(tail[:120]) >= 3:
            return True
    return False


def _read_memory() -> str:
    """Lê o arquivo de memória permanente, criando-o se não existir."""
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(
            "# MEMÓRIA PERMANENTE DO AGENTE\n"
            "- Regra de Versionamento: Nunca alterar main.py ou arquivos principais diretamente; "
            "criar main_2.py, main_3.py, etc., e comparar com o original para garantir as funções.\n",
            encoding="utf-8",
        )
    return MEMORY_FILE.read_text(encoding="utf-8", errors="replace")


# =====================================================================
# 1. DEFINIÇÃO DAS FERRAMENTAS DO AGENTE (TODAS PRESERVADAS)
# =====================================================================


def listar_arquivos(diretorio: str = ".") -> str:
    """Lista arquivos e subpastas dentro de um diretório do projeto."""
    try:
        target = _resolve_safe_path(diretorio)
        if not target.exists():
            return f"Erro: diretório '{diretorio}' não encontrado."
        itens = []
        for entry in sorted(target.iterdir()):
            if entry.name.startswith(".venv") or entry.name == "__pycache__":
                continue
            tipo = "[DIR]" if entry.is_dir() else "[ARQ]"
            itens.append(f"{tipo} {entry.relative_to(WORKSPACE)}")
        return "\n".join(itens) if itens else "(Diretório vazio)"
    except Exception as e:
        return f"Erro ao listar arquivos: {e}"


def ler_arquivo_texto(caminho: str) -> str:
    """Lê o conteúdo de um arquivo de texto ou código do projeto."""
    try:
        target = _resolve_safe_path(caminho)
        if not target.is_file():
            return f"Erro: arquivo '{caminho}' não existe."
        conteudo = target.read_text(encoding="utf-8", errors="replace")
        if len(conteudo) > 20000:
            return conteudo[:20000] + "\n...[TRUNCADO POR TAMANHO]..."
        return conteudo
    except Exception as e:
        return f"Erro ao ler arquivo: {e}"


def escrever_arquivo(caminho: str, conteudo: str) -> str:
    """Cria ou sobrescreve um arquivo de texto/código no projeto."""
    try:
        target = _resolve_safe_path(caminho)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(conteudo, encoding="utf-8")
        return f"Sucesso: arquivo '{caminho}' salvo ({len(conteudo)} caracteres)."
    except Exception as e:
        return f"Erro ao escrever arquivo: {e}"


def ler_imagem(caminho: str) -> str:
    """Carrega uma imagem (PNG, JPG, WEBP) do projeto para o modelo analisar visualmente."""
    try:
        target = _resolve_safe_path(caminho)
        if not target.is_file():
            return f"Erro: imagem '{caminho}' não encontrada."
        return f"__IMAGE_READY__:{target}"
    except Exception as e:
        return f"Erro ao acessar imagem: {e}"


def buscar_web(query: str, num_resultados: int = 5) -> str:
    """Busca informações na web sem exigir chave de API ou faturamento no Google Cloud."""
    q = query.strip()
    if not q:
        return "Erro: consulta de busca vazia."

    limit = max(1, min(int(num_resultados), 10))

    try:
        resultados = DDGS().text(q, region="br-pt", max_results=limit)
        if not resultados:
            return f"Nenhum resultado encontrado para: '{q}'."

        linhas = [f"Resultados para '{q}':"]
        for idx, item in enumerate(resultados, start=1):
            titulo = str(item.get("title", "Sem título")).strip()
            link = str(item.get("href", "")).strip()
            trecho = str(item.get("body", "")).replace("\n", " ").strip()
            linhas.append(f"{idx}. {titulo}\n   URL: {link}\n   Resumo: {trecho}")

        return "\n\n".join(linhas)[:15000]
    except Exception as e:
        return f"Erro ao executar busca web: {e}"


def ler_pagina_web(url: str) -> str:
    """Acessa uma URL (http/https) e extrai o texto visível da página."""
    if not url.startswith(("http://", "https://")):
        return "Erro: a URL deve começar com http:// ou https://"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")

        limpo = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE
        )
        limpo = re.sub(r"<[^>]+>", " ", limpo)
        limpo = html.unescape(limpo)
        limpo = re.sub(r"\s+", " ", limpo).strip()

        if not limpo:
            return "Aviso: a página retornou vazia ou exige JavaScript para renderizar o conteúdo."

        return limpo[:6000] + ("\n...[TRUNCADO]..." if len(limpo) > 6000 else "")
    except Exception as e:
        return f"Erro ao acessar a página '{url}': {e}"


def executar_comando(comando: str) -> str:
    """Executa um comando no terminal Linux dentro da pasta do projeto."""
    try:
        tokens = shlex.split(comando)
    except ValueError as e:
        return f"Erro de sintaxe no comando: {e}"

    if not tokens:
        return "Erro: comando vazio."

    operadores_encadeados = {"&&", "||", ";", "|", ">", ">>", "<", "$(", "`"}
    tem_operador = any(op in comando for op in operadores_encadeados)
    binario_base = os.path.basename(tokens[0])

    autorizado = (binario_base in ALLOWED_COMMANDS) and not tem_operador

    if not autorizado:
        print(f"\n⚠️  [PERMISSÃO NECESSÁRIA] O agente quer executar:\n   $ {comando}")
        escolha = (
            input("Permitir? [s]im / [n]ão / [a]dicionar comando à lista permanente: ")
            .strip()
            .lower()
        )

        if escolha == "a":
            if tem_operador:
                print(
                    "   (Comandos encadeados não podem ser salvos na lista automática; permitido apenas desta vez.)"
                )
            else:
                ALLOWED_COMMANDS.add(binario_base)
                save_allowlist(ALLOWED_COMMANDS)
                print(
                    f"   ✅ '{binario_base}' adicionado à lista permanente ({ALLOWLIST_FILE.name})."
                )
        elif escolha != "s":
            return "Execução negada pelo usuário."

    try:
        proc = subprocess.run(
            comando,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=120,
        )
        saida = ""
        if proc.stdout:
            saida += f"STDOUT:\n{proc.stdout}\n"
        if proc.stderr:
            saida += f"STDERR:\n{proc.stderr}\n"
        saida += f"(Exit code: {proc.returncode})"
        return saida[:15000]
    except subprocess.TimeoutExpired:
        return "Erro: o comando excedeu o tempo limite de 120 segundos."
    except Exception as e:
        return f"Erro ao executar comando: {e}"


def gerar_imagem(
    prompt: str,
    negative_prompt: str = (
        "extra head, extra heads, duplicate face, extra limbs, extra arms, "
        "extra legs, mutated hands, bad anatomy, disfigured, blurry, low quality, watermark"
    ),
    width: int = 640,   # [NOVO v3] Padrão reduzido de 768 para 640 para evitar pico de RAM/VRAM
    height: int = 640,
    steps: int = 22,
    guidance_scale: float = 5.5,
    seed: int | None = None,
) -> str:
    """Desativa o LLM, valida RAM disponível, carrega o modelo de difusão com proteção de memória, gera e entrega a imagem."""
    if not torch.cuda.is_available():
        return "Erro: CUDA não disponível para gerar imagem."

    pipe = None
    try:
        print(f"\n🔄 [Memória] Desativando LLM ({MODEL}) e limpando cache...")
        _prune_old_images_from_history(keep_last_n_msgs=0)
        _unload_llm(MODEL)
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        # [NOVO v3] Trava de segurança: aborta antes de congelar o Linux se a RAM livre estiver crítica
        ram_livre = _get_available_ram_gb()
        print(f"📊 [Memória] RAM disponível após desligar LLM: {ram_livre:.2f} GB")
        if ram_livre < 4.5:
            return (
                f"Erro preventivo: apenas {ram_livre:.1f} GB de RAM livre no sistema. "
                "Feche abas pesadas do navegador ou outros programas antes de gerar imagens para não travar o PC."
            )

        # Limita dimensões máximas para não estourar VRAM/RAM
        width = min(int(width), 768)
        height = min(int(height), 768)

        print(f"🎨 [Artista] Carregando '{DIFFUSION_MODEL_ID}'...")
        OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
        pipe = DiffusionPipeline.from_pretrained(
            DIFFUSION_MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            variant="fp16",
        )

        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )
        except Exception:
            pass

        mem_mode = os.getenv("DIFFUSION_MEM_MODE", "seq_offload").lower()
        if mem_mode == "gpu":
            pipe.to("cuda")
        elif mem_mode == "model_offload":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")

        generator = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )

        print(
            f'🖌️  [Artista] Gerando ({width}x{height}, steps={steps}, cfg={guidance_scale}, seed={seed}): "{prompt}"'
        )
        # [NOVO v3] inference_mode evita alocação de grafo de gradientes e economiza RAM/VRAM
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{_sanitize_filename(prompt)}.png"
        out_path = OUTPUT_DIR / filename
        result.images[0].save(out_path)

        rel_path = out_path.relative_to(WORKSPACE)
        print(f"✅ [Artista] Imagem salva em: {rel_path}")
        # [NOVO v3] Prefixo especial para o servidor web renderizar a imagem na tela
        return f"__GENERATED_IMAGE__:{rel_path}"

    except torch.OutOfMemoryError:
        return "Erro: estouro de memória (OOM). Tente reduzir resolução (ex: 512x512) ou fechar apps pesados."
    except Exception as e:
        return f"Erro ao gerar imagem: {e}"
    finally:
        print("🧹 [Artista] Desligando artista e liberando memória...")
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        print("🤖 [Memória] Artista desligada; LLM será reativado na próxima chamada.")


def iniciar_servidor_web(porta: int = 5000, abrir_navegador: bool = True) -> str:
    """Inicia o servidor web local de chat em segundo plano com suporte a Ctrl+V, Drag & Drop e histórico."""
    global _server_started
    if _server_started:
        return f"O servidor web já está ativo em http://127.0.0.1:{porta}"

    def _run():
        app.run(host="127.0.0.1", port=porta, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _server_started = True
    url = f"http://127.0.0.1:{porta}"
    if abrir_navegador:
        webbrowser.open(url)
    return f"Servidor web iniciado com sucesso em {url}"


AVAILABLE_TOOLS = {
    "listar_arquivos": listar_arquivos,
    "ler_arquivo_texto": ler_arquivo_texto,
    "escrever_arquivo": escrever_arquivo,
    "ler_imagem": ler_imagem,
    "buscar_web": buscar_web,
    "executar_comando": executar_comando,
    "gerar_imagem": gerar_imagem,
    "ler_pagina_web": ler_pagina_web,
    "iniciar_servidor_web": iniciar_servidor_web,
}


# =====================================================================
# 2. GERENCIAMENTO DE HISTÓRICO E LOOP DO AGENTE (STREAMING + ANTI-LOOP)
# =====================================================================


def _build_system_prompt() -> str:
    memoria_atual = _read_memory()
    return f"""Você é um agente de desenvolvimento e automação local operando no diretório: {WORKSPACE}.
Você tem acesso a ferramentas para listar arquivos, ler textos/códigos, ler imagens, escrever arquivos, buscar informações na web (buscar_web), ler páginas web (ler_pagina_web), executar comandos no terminal, gerar imagens (gerar_imagem) e iniciar o servidor web de chat (iniciar_servidor_web).

MEMÓRIA PERMANENTE (leia e respeite sempre):
{memoria_atual}

Regras:
1. Sempre inspecione os arquivos antes de fazer suposições sobre o projeto.
2. Nunca sobrescreva `main.py` ou arquivos principais diretamente: crie versões incrementais (`main_2.py`, `main_3.py`, etc.) e compare para garantir integridade.
3. Ao usar `gerar_imagem`, prefira resoluções entre 512x512 e 640x640 para estabilidade de memória. Quando a imagem for gerada, apenas confirme ao usuário que ela já está exibida na tela.
4. Responda sempre em português de forma clara e objetiva."""


history = [{"role": "system", "content": _build_system_prompt()}]
_server_started = False


def stream_agent_turn(user_prompt: str, user_images: list[str] | None = None, max_steps: int = 8):
    """Gerador que executa o turno do agente em streaming, emitindo eventos de pensamento, ferramentas, imagens e texto."""
    history[0]["content"] = _build_system_prompt()
    _prune_old_images_from_history(keep_last_n_msgs=1)

    user_msg = {"role": "user", "content": user_prompt}
    if user_images:
        user_msg["images"] = user_images
    history.append(user_msg)

    use_think = True

    for _ in range(max_steps):
        try:
            stream = ollama.chat(
                model=MODEL,
                messages=history,
                tools=list(AVAILABLE_TOOLS.values()),
                think=use_think,
                stream=True,
                options={
                    "num_ctx": 12288,   # [NOVO v3] Reduzido levemente de 16384 para 12288 para folga de RAM
                    "num_predict": 4096,
                    "temperature": 0.2,
                },
            )
        except Exception as e:
            yield {"type": "error", "content": f"Falha na conexão com Ollama: {e}"}
            return

        full_thought = ""
        full_content = ""
        tool_calls = []
        loop_aborted = False

        for chunk in stream:
            msg = chunk.message
            t_part = getattr(msg, "thinking", None) or ""
            c_part = msg.content or ""

            if t_part:
                full_thought += t_part
                yield {"type": "thinking", "content": t_part}

                if len(full_thought) > 6000 or _has_thinking_loop(full_thought):
                    yield {
                        "type": "thinking",
                        "content": "\n[⚠️ Pensamento interrompido por limite/repetição; retomando execução...]\n",
                    }
                    use_think = False
                    loop_aborted = True
                    break

            if c_part:
                full_content += c_part
                yield {"type": "content", "content": c_part}

            if msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        if loop_aborted:
            continue

        assistant_record = {"role": "assistant", "content": full_content}
        if full_thought:
            assistant_record["thinking"] = full_thought
        if tool_calls:
            assistant_record["tool_calls"] = tool_calls
        history.append(assistant_record)

        if not tool_calls:
            break

        pending_images = []
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments
            yield {"type": "tool", "content": f"{fn_name}({fn_args})"}

            func = AVAILABLE_TOOLS.get(fn_name)
            if not func:
                result = f"Erro: ferramenta '{fn_name}' desconhecida."
            else:
                result = func(**fn_args)

            if isinstance(result, str) and result.startswith("__IMAGE_READY__:"):
                img_path = Path(result.split(":", 1)[1])
                img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                pending_images.append(img_b64)
                tool_output = f"Imagem '{img_path.name}' carregada com sucesso."
            # [NOVO v3] Intercepta a imagem gerada pelo SDXL e envia a URL para o chat web exibir
            elif isinstance(result, str) and result.startswith("__GENERATED_IMAGE__:"):
                rel_path = result.split(":", 1)[1].replace("\\", "/")
                yield {"type": "image", "url": f"/{rel_path}"}
                tool_output = f"Imagem gerada com sucesso e já exibida na interface do usuário em '{rel_path}'."
            else:
                tool_output = str(result)

            history.append({
                "role": "tool",
                "tool_name": fn_name,
                "content": tool_output,
            })

        if pending_images:
            history.append({
                "role": "user",
                "content": "Aqui está a imagem solicitada pela ferramenta:",
                "images": pending_images,
            })


def run_agent_turn(user_prompt: str, max_steps: int = 8) -> None:
    """Executa o turno do agente no terminal acompanhando a saída em tempo real."""
    in_thinking = False
    in_content = False

    for event in stream_agent_turn(user_prompt, max_steps=max_steps):
        etype = event["type"]

        if etype == "thinking":
            if not in_thinking:
                print("\n[Pensando]: ", end="", flush=True)
                in_thinking = True
            print(event["content"], end="", flush=True)
        elif etype == "tool":
            if in_thinking:
                print("\n")
                in_thinking = False
            print(f"\n🔧 [Tool] {event['content']}", flush=True)
        elif etype == "image":
            print(f"\n🖼️  [Imagem exibida no chat]: {event['url']}", flush=True)
        elif etype == "content":
            if in_thinking:
                print("\n")
                in_thinking = False
            if not in_content:
                print("\nAgente: ", end="", flush=True)
                in_content = True
            print(event["content"], end="", flush=True)
        elif etype == "error":
            print(f"\n❌ {event['content']}", flush=True)

    print("\n")


# =====================================================================
# 3. SERVIDOR WEB FLASK (ESTILO GEMINI / CHATGPT / CLAUDE)
# =====================================================================

app = Flask(__name__, template_folder=str(WORKSPACE / "templates"))


@app.route("/")
def index():
    return render_template("index.html")


# [NOVO v3] Rota que permite ao navegador carregar as imagens geradas na pasta outputs/
@app.route("/outputs/<path:filename>")
def serve_output_image(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/clear", methods=["POST"])
def clear_chat():
    global history
    history = [{"role": "system", "content": _build_system_prompt()}]
    gc.collect()
    return jsonify({"status": "ok"})


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    prompt_text = request.form.get("prompt", "").strip()
    uploaded_files = request.files.getlist("files")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    extracted_texts = []
    images_b64 = []

    for f in uploaded_files:
        if not f.filename:
            continue
        safe_name = f"{datetime.now().strftime('%H%M%S')}_{Path(f.filename).name}"
        saved_path = UPLOADS_DIR / safe_name
        f.save(saved_path)

        ext = saved_path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            images_b64.append(base64.b64encode(saved_path.read_bytes()).decode("utf-8"))
            extracted_texts.append(f"[Imagem anexada salva em: uploads/{safe_name}]")
        else:
            try:
                content = saved_path.read_text(encoding="utf-8", errors="replace")
                extracted_texts.append(
                    f"--- Conteúdo do arquivo '{f.filename}' (salvo em uploads/{safe_name}) ---\n"
                    f"{content[:15000]}\n--- Fim do arquivo ---"
                )
            except Exception as e:
                extracted_texts.append(f"[Arquivo binário anexado: uploads/{safe_name} ({e})]")

    full_prompt = prompt_text
    if extracted_texts:
        anexos_str = "\n\n".join(extracted_texts)
        full_prompt = f"{prompt_text}\n\n{anexos_str}".strip()

    if not full_prompt and not images_b64:
        return jsonify({"error": "Entrada vazia"}), 400

    def generate():
        for event in stream_agent_turn(full_prompt, user_images=images_b64 or None):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


def main():
    if "--web" in sys.argv:
        print("🌐 Iniciando GabyBOT Web (v3) em http://127.0.0.1:5000 ...")
        webbrowser.open("http://127.0.0.1:5000")
        app.run(host="127.0.0.1", port=5000, debug=False)
        return

    print(f"🤖 Agente criativo (v3) iniciado em: {WORKSPACE}")
    print(f"📋 Comandos auto-aprovados: {sorted(ALLOWED_COMMANDS)}")
    print("💡 Digite 'servidor' para abrir a interface web ou 'sair' para encerrar.\n")

    while True:
        try:
            user_input = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

        if not user_input:
            continue
        if user_input.lower() in {"sair", "exit", "quit"}:
            break
        if user_input.lower() == "servidor":
            print(iniciar_servidor_web())
            continue

        run_agent_turn(user_input)


if __name__ == "__main__":
    main()