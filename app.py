
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Modèle chargé une seule fois
model = None
tokenizer = None
loaded_model_name = None


def load_model(model_name):
    global model, tokenizer, loaded_model_name

    if model is None or loaded_model_name != model_name:
        print(f"Chargement du modèle : {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        loaded_model_name = model_name

    return tokenizer, model


def generate_text(model_name, prompt, temperature, top_k, top_p):

    if not prompt.strip():
        return "Veuillez saisir un prompt."

    try:
        tokenizer, model = load_model(model_name)

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature,
                top_k=int(top_k),
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id
            )

        result = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return result

    except Exception as e:
        return f"Erreur : {str(e)}"


# Interface Gradio
with gr.Blocks(title="TP3 - GenAI") as demo:

    gr.Markdown("# TP N°3 : Analyse des paramètres de génération des LLMs")

    gr.Markdown(
        "Étudier l'influence de Temperature, Top-k et Top-p "
        "sur la génération de texte."
    )

    model_name = gr.Textbox(
        label="Model Name",
        value="gpt2"
    )

    prompt = gr.Textbox(
        label="Prompt",
        value="Explique l'intelligence artificielle à un étudiant de licence en 5 phrases",
        lines=3
    )

    temperature = gr.Slider(
        minimum=0.1,
        maximum=2.0,
        value=0.8,
        step=0.1,
        label="Temperature"
    )

    top_k = gr.Slider(
        minimum=1,
        maximum=100,
        value=50,
        step=1,
        label="Top-k"
    )

    top_p = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.9,
        step=0.05,
        label="Top-p"
    )

    generate_button = gr.Button(
        "Generate Text",
        variant="primary"
    )

    output = gr.Textbox(
        label="Generated Text",
        lines=10
    )

    generate_button.click(
        fn=generate_text,
        inputs=[
            model_name,
            prompt,
            temperature,
            top_k,
            top_p
        ],
        outputs=output
    )


if __name__ == "__main__":
    demo.launch()
