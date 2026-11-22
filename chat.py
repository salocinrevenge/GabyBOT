# Teste: ollama run gemma4:12b "Responda em uma frase: o que é uma GPU?"
import ollama

MODEL = "gemma4:12b"
USE_THINKING = True  # Mude para False para respostas imediatas sem raciocínio

history = [
    {
        "role": "system",
        "content": "Seja objetivo. Se o modo de raciocínio estiver ativo, mantenha os passos de pensamento concisos.",
    }
]


def ask_model(prompt: str) -> str:
    history.append({"role": "user", "content": prompt})

    response_chunks = ollama.chat(
        model=MODEL,
        messages=history,
        stream=True,
        think=USE_THINKING,  # Ativa ou desativa o bloco de raciocínio
        options={
            "num_ctx": 8192,      # Tamanho máximo da memória de contexto (entrada + saída)
            "num_predict": 2048,  # Limite máximo de tokens gerados (thinking + resposta final)
            "temperature": 0.7, # 0 a 0.3: objetivo, preciso. 0.7 a 1.0: criativo
        },
    )

    full_reply = []
    is_thinking = False

    for chunk in response_chunks:
        msg = chunk.get("message", {})

        # 1. Captura e exibe o raciocínio (se think=True)
        thought = msg.get("thinking", "")
        if thought:
            if not is_thinking:
                print("\n[Pensando...]\n", end="", flush=True)
                is_thinking = True
            print(thought, end="", flush=True)

        # 2. Captura e exibe a resposta final
        content = msg.get("content", "")
        if content:
            if is_thinking:
                print("\n\n[Resposta]\n", end="", flush=True)
                is_thinking = False
            print(content, end="", flush=True)
            full_reply.append(content)

    print()
    reply_text = "".join(full_reply)
    # Salvamos apenas a resposta final no histórico para não estourar o num_ctx com pensamentos antigos
    history.append({"role": "assistant", "content": reply_text})
    return reply_text


def main():
    print(f"Conectado ao Ollama ({MODEL}). Digite 'sair' para encerrar.\n")
    while True:
        try:
            user_input = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

        if not user_input:
            continue

        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Encerrando...")
            break

        print("IA: ", end="", flush=True)
        ask_model(user_input)


if __name__ == "__main__":
    main()

# Após, usar ollama stop gemma4:12b

