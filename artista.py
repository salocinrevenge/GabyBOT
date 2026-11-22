import os
import re
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

# Modelo leve padrão para proteger os 16 GB de RAM do sistema;
# você pode sobrescrever com DIFFUSION_MODEL_ID=<outro_modelo>
MODEL_ID = os.getenv("DIFFUSION_MODEL_ID", "stabilityai/sdxl-turbo")
OUTPUT_DIR = Path("outputs")


def sanitize_filename(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower()).strip("_")
    return slug[:max_len] or "desenho"


def load_artist(model_id: str) -> DiffusionPipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Verifique o driver NVIDIA e o PyTorch.")

    print(f"Carregando artista ({model_id})...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    # Por padrão, mantém na GPU (16 GB VRAM) para não estourar os 16 GB de RAM do sistema.
    # Se quiser forçar CPU offload para modelos maiores, rode com DIFFUSION_CPU_OFFLOAD=1.
    if os.getenv("DIFFUSION_CPU_OFFLOAD", "0") == "1":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    return pipe


def draw(
    pipe: DiffusionPipeline,
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    image = result.images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{sanitize_filename(prompt)}.png"
    out_path = OUTPUT_DIR / filename
    image.save(out_path)

    torch.cuda.empty_cache()
    return out_path


def parse_command(raw: str) -> tuple[str, int, int, int, int | None]:
    """
    Aceita parâmetros após '|' separados por espaço ou por '|'.
    Ex:
      ursinho cor de lavanda | w=512 h=512 steps=4 seed=1
    """
    width, height, steps, seed = 512, 512, 4, None

    if "|" not in raw:
        return raw.strip(), width, height, steps, seed

    prompt_part, params_part = raw.split("|", 1)
    prompt = prompt_part.strip()
    tokens = params_part.replace("|", " ").split()

    for token in tokens:
        if "=" not in token:
            continue
        k, v = [x.strip().lower() for x in token.split("=", 1)]
        try:
            if k in {"w", "width"}:
                width = int(v)
            elif k in {"h", "height"}:
                height = int(v)
            elif k in {"steps", "n_steps"}:
                steps = int(v)
            elif k == "seed":
                seed = int(v)
        except ValueError:
            print(f"⚠️  Valor inválido para '{k}': '{v}' (mantendo padrão).")

    return prompt, width, height, steps, seed


def main() -> None:
    pipe = load_artist(MODEL_ID)
    print("\n🎨 Artista pronta!")
    print("Digite o que você quer desenhar.")
    print("Opcional: use '|' para ajustar tamanho/passos/seed.")
    print("Exemplo: Ursinho cor de lavanda andando de bicicleta no arco iris | w=512 h=512 steps=4 seed=1")
    print("Digite 'sair' para encerrar.\n")

    while True:
        try:
            raw = input("Mandar desenhar: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando artista...")
            break

        if not raw:
            continue
        if raw.lower() in {"sair", "exit", "quit"}:
            print("Encerrando artista...")
            break

        prompt, width, height, steps, seed = parse_command(raw)
        print(f"🖌️  Gerando ({width}x{height}, {steps} passos, seed={seed})...")

        try:
            saved_path = draw(
                pipe=pipe,
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
            )
            print(f"✅ Salvo em: {saved_path}\n")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("❌ Estouro de memória (OOM). Tente reduzir resolução ou fechar apps pesados.\n")
        except Exception as e:
            print(f"❌ Erro durante a geração: {e}\n")


if __name__ == "__main__":
    main()
    #Lavender-colored rabbit riding a bicycle on a rainbow between two planets, in a watercolor style | w=512 h=512 steps=16 seed=2
