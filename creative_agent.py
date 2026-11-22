import base64
import gc
import json
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import ollama
import torch
from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline

MODEL = "gemma4:12b"
# Modelo base SDXL completo: mais lento que Turbo/Lightning, porém com CFG (guidance_scale)
# e mais passos para melhor coerência e detalhes sem exigir modelos de poucos passos.
DIFFUSION_MODEL_ID = os.getenv(
    "DIFFUSION_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"
)
WORKSPACE = Path.cwd().resolve()
OUTPUT_DIR = WORKSPACE / "outputs"
ALLOWLIST_FILE = WORKSPACE / ".agent_allowed_commands.json"

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


def _has_thinking_loop(thought: str | None) -> bool:
    """Detecta se o bloco de raciocínio ficou repetitivo."""
    if not thought:
        return False
    lines = [ln.strip() for ln in thought.splitlines() if ln.strip()]
    if len(lines) < 8:
        return False
    unique_ratio = len(set(lines)) / len(lines)
    return unique_ratio < 0.35


# =====================================================================
# 1. DEFINIÇÃO DAS FERRAMENTAS DO AGENTE
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
            input(
                "Permitir? [s]im / [n]ão / [a]dicionar comando à lista permanente: "
            )
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
    width: int = 768,
    height: int = 768,
    steps: int = 25,
    guidance_scale: float = 5.5,
    seed: int | None = None,
) -> str:
    """Desativa o LLM, carrega o modelo de difusão com foco em qualidade e baixo uso de RAM, gera a imagem e desliga."""
    if not torch.cuda.is_available():
        return "Erro: CUDA não disponível para gerar imagem."

    pipe = None
    try:
        print(f"\n🔄 [Memória] Desativando LLM ({MODEL}) para liberar memória...")
        _unload_llm(MODEL)
        gc.collect()
        torch.cuda.empty_cache()

        print(f"🎨 [Artista] Carregando '{DIFFUSION_MODEL_ID}'...")
        pipe = DiffusionPipeline.from_pretrained(
            DIFFUSION_MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            variant="fp16",
        )

        # Scheduler mais estável para qualidade em ~20-30 passos
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )
        except Exception:
            pass

        # Modo padrão: sequential offload reduz picos de RAM em 16 GB (mais lento, mais seguro).
        # Para priorizar velocidade se tiver RAM sobrando: DIFFUSION_MEM_MODE=gpu
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
            pipe.enable_attention_slicing()

        generator = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )

        print(
            f'🖌️  [Artista] Gerando ({width}x{height}, steps={steps}, cfg={guidance_scale}, seed={seed}): "{prompt}"'
        )
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
        return f"Imagem gerada com sucesso e salva em '{rel_path}'."

    except torch.OutOfMemoryError:
        return "Erro: estouro de memória (OOM). Tente reduzir resolução (ex: 512x512 ou 640x640) ou fechar apps pesados."
    except Exception as e:
        return f"Erro ao gerar imagem: {e}"
    finally:
        print("🧹 [Artista] Desligando artista e liberando memória...")
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🤖 [Memória] Artista desligada; LLM será reativado na próxima chamada.")


AVAILABLE_TOOLS = {
    "listar_arquivos": listar_arquivos,
    "ler_arquivo_texto": ler_arquivo_texto,
    "escrever_arquivo": escrever_arquivo,
    "ler_imagem": ler_imagem,
    "executar_comando": executar_comando,
    "gerar_imagem": gerar_imagem,
}


# =====================================================================
# 2. LOOP PRINCIPAL DO AGENTE (ReAct)
# =====================================================================

SYSTEM_PROMPT = f"""Você é um agente de desenvolvimento e automação local operando no diretório: {WORKSPACE}.
Você tem acesso a ferramentas para listar arquivos, ler textos/códigos, ler imagens, escrever arquivos, executar comandos no terminal e gerar imagens por difusão (gerar_imagem).
Regras:
1. Sempre inspecione os arquivos antes de fazer suposições sobre o projeto.
2. Ao escrever arquivos, forneça o conteúdo completo e funcional.
3. Ao ler uma imagem, identifique o que ela mostra de forma direta e objetiva; não repita análises nem fique em dúvida em loop.
4. Ao usar gerar_imagem, chame a ferramenta uma vez com um prompt detalhado em inglês descrevendo cena, iluminação, estilo e anatomia explícita (ex: "single character, one head, two arms, two legs").
5. Se uma ferramenta retornar erro, não fique repetindo a mesma chamada indefinidamente: ajuste uma vez se fizer sentido ou explique o erro ao usuário.
6. Responda sempre em português de forma clara e objetiva."""

history = [{"role": "system", "content": SYSTEM_PROMPT}]


def run_agent_turn(user_prompt: str, max_steps: int = 8) -> None:
    history.append({"role": "user", "content": user_prompt})
    use_think = True

    for _ in range(max_steps):
        response = ollama.chat(
            model=MODEL,
            messages=history,
            tools=list(AVAILABLE_TOOLS.values()),
            think=use_think,
            options={
                "num_ctx": 16384,
                "num_predict": 1536,
                "temperature": 0.2,
            },
        )

        msg = response.message
        thought = getattr(msg, "thinking", None)

        if _has_thinking_loop(thought):
            print("\n⚠️  [Aviso] Loop detectado no raciocínio; tentando próxima etapa sem thinking...\n")
            use_think = False
            continue

        history.append(msg)

        if thought:
            print(f"\n[Pensando]: {thought}\n")

        if not msg.tool_calls:
            print(f"\nAgente: {msg.content}\n")
            break

        pending_images = []
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments
            print(f"🔧 [Tool] {fn_name}({fn_args})")

            func = AVAILABLE_TOOLS.get(fn_name)
            if not func:
                result = f"Erro: ferramenta '{fn_name}' desconhecida."
            else:
                result = func(**fn_args)

            if isinstance(result, str) and result.startswith("__IMAGE_READY__:"):
                img_path = Path(result.split(":", 1)[1])
                img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                pending_images.append(img_b64)
                tool_output = f"Imagem '{img_path.name}' carregada com sucesso. Analise a imagem anexada abaixo."
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


def main():
    print(f"🤖 Agente criativo iniciado em: {WORKSPACE}")
    print(f"📋 Comandos auto-aprovados: {sorted(ALLOWED_COMMANDS)}")
    print("Digite 'sair' para encerrar.\n")

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

        run_agent_turn(user_input)


if __name__ == "__main__":
    main()
