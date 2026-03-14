import os
import gradio as gr
from huggingface_hub import InferenceClient

TOKEN_FILE = "token.txt"
QUALITE = "ultra-realistic, high quality, 4k, detailed, sharp focus"
NEGATIF = "low quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, extra limbs, close up, b&w, weird colors"


def charger_token() -> str:
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def sauvegarder_token(token: str):
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(token.strip())


def generer_image(prompt: str, token: str):
    if not token.strip():
        return None, "❌ Veuillez entrer votre token Hugging Face."
    if not prompt.strip():
        return None, "❌ Veuillez décrire l'image à générer."

    sauvegarder_token(token)

    prompt_final = f"{prompt}, {QUALITE}"

    try:
        client = InferenceClient(
            provider="auto",
            api_key=token.strip(),
        )

        image = client.text_to_image(
            prompt=prompt_final,
            model="stabilityai/stable-diffusion-xl-base-1.0",
            negative_prompt=NEGATIF,
            guidance_scale=9.0,
            num_inference_steps=50,
            width=1024,
            height=1024,
        )

        image_path = "image_generee.png"
        image.save(image_path)
        return image_path, "✅ Image générée avec succès !"

    except Exception as e:
        return None, f"❌ Erreur : {e}"


with gr.Blocks(title="Générateur d'Images Réalistes") as app:

    gr.Markdown("# 🎨 Générateur d'Images Réalistes — Hugging Face")
    gr.Markdown("📦 Modèle : `stabilityai/stable-diffusion-xl-base-1.0`")

    token_input = gr.Textbox(
        label="🔑 Token Hugging Face",
        placeholder="hf_xxxxxxxxxxxxxxxx",
        type="password",
        value=charger_token()
    )

    prompt_input = gr.Textbox(
        label="✏️ Décrivez l'image en français",
        placeholder="un data analyst",
        lines=3
    )

    generate_btn = gr.Button("🎨 Générer l'image", variant="primary")

    with gr.Row():
        image_output = gr.Image(label="Image générée")
        status_output = gr.Textbox(label="Statut", interactive=False)

    generate_btn.click(
        fn=generer_image,
        inputs=[prompt_input, token_input],
        outputs=[image_output, status_output]
    )

app.launch()