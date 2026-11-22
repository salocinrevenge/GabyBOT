import base64
import json
import os
import shlex
import subprocess
from pathlib import Path
import ollama

MODEL = "gemma4:12b"
WORKSPACE = Path.cwd().resolve()  # Restringe leitura/escrita à pasta atual por segurança
ALLOWLIST_FILE = WORKSPACE / ".agent_allowed_commands.json"

# Comandos seguros iniciais (apenas o executável base, ex: "ls", "cat", "git")
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
        # Retorna um marcador especial que o loop principal usará para injetar a imagem na conversa
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

    # Pega o executável principal (ex: 'python3' de 'python3 script.py')
    # Bloqueia operadores de encadeamento perigosos na verificação automática
    operadores_encadeados = {"&&", "||", ";", "|", ">", ">>", "<", "$(", "`"}
    tem_operador = any(op in comando for op in operadores_encadeados)
    binario_base = os.path.basename(tokens[0])

    autorizado = (binario_base in ALLOWED_COMMANDS) and not tem_operador

    if not autorizado:
        print(f"\n⚠️  [PERMISSÃO NECESSÁRIA] O agente quer executar:\n   $ {comando}")
        escolha = input("Permitir? [s]im / [n]ão / [a]dicionar comando à lista permanente: ").strip().lower()

        if escolha == "a":
            if tem_operador:
                print("   (Comandos encadeados não podem ser salvos na lista automática; permitido apenas desta vez.)")
            else:
                ALLOWED_COMMANDS.add(binario_base)
                save_allowlist(ALLOWED_COMMANDS)
                print(f"   ✅ '{binario_base}' adicionado à lista permanente ({ALLOWLIST_FILE.name}).")
        elif escolha != "s":
            return "Execução negada pelo usuário."

    try:
        proc = subprocess.run(
            comando,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=60,
        )
        saida = ""
        if proc.stdout:
            saida += f"STDOUT:\n{proc.stdout}\n"
        if proc.stderr:
            saida += f"STDERR:\n{proc.stderr}\n"
        saida += f"(Exit code: {proc.returncode})"
        return saida[:15000]  # Evita estourar o contexto com logs gigantes
    except subprocess.TimeoutExpired:
        return "Erro: o comando excedeu o tempo limite de 60 segundos."
    except Exception as e:
        return f"Erro ao executar comando: {e}"


# Mapeamento nome -> função Python
AVAILABLE_TOOLS = {
    "listar_arquivos": listar_arquivos,
    "ler_arquivo_texto": ler_arquivo_texto,
    "escrever_arquivo": escrever_arquivo,
    "ler_imagem": ler_imagem,
    "executar_comando": executar_comando,
}


# =====================================================================
# 2. LOOP PRINCIPAL DO AGENTE (ReAct)
# =====================================================================

SYSTEM_PROMPT = f"""Você é um agente de desenvolvimento e automação local operando no diretório: {WORKSPACE}.
Você tem acesso a ferramentas para listar arquivos, ler textos/códigos, ler imagens, escrever arquivos e executar comandos no terminal.
Regras:
1. Sempre inspecione os arquivos antes de fazer suposições sobre o projeto.
2. Ao escrever arquivos, forneça o conteúdo completo e funcional.
3. Responda sempre em português de forma clara e objetiva."""

history = [{"role": "system", "content": SYSTEM_PROMPT}]


def run_agent_turn(user_prompt: str) -> None:
    history.append({"role": "user", "content": user_prompt})

    # Loop interno: o agente pode chamar múltiplas ferramentas em sequência antes de dar a resposta final
    while True:
        response = ollama.chat(
            model=MODEL,
            messages=history,
            tools=list(AVAILABLE_TOOLS.values()),  # O SDK lê docstrings e type hints automaticamente!
            think=True,
            options={
                "num_ctx": 16384,     # Um pouco maior para caber código + saídas de ferramentas
                "temperature": 0.2,   # Baixa temperatura deixa o uso de ferramentas mais preciso
            },
        )

        msg = response.message
        history.append(msg)  # Salva a resposta do assistente (incluindo tool_calls, se houver)

        # Exibe raciocínio se presente
        if getattr(msg, "thinking", None):
            print(f"\n[Pensando]: {msg.thinking}\n")

        # Se o modelo NÃO pediu nenhuma ferramenta, ele terminou o turno!
        if not msg.tool_calls:
            print(f"\nAgente: {msg.content}\n")
            break

        # Executa cada ferramenta solicitada pelo modelo
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

            # Tratamento especial para imagens: injeta na visão do Gemma 4
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

        # Se alguma ferramenta carregou uma imagem, envia logo em seguida para o encoder visual do Gemma 4
        if pending_images:
            history.append({
                "role": "user",
                "content": "Aqui está a imagem solicitada pela ferramenta:",
                "images": pending_images,
            })


def main():
    print(f"🤖 Agente iniciado em: {WORKSPACE}")
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
