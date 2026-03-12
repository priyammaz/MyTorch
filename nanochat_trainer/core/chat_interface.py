"""
Gradio Chat Interface for NanoChat GPT
"""
import os
import yaml
import argparse
import gradio as gr
import mytorch
from .nanochat_gpt import GPT, GPTConfig
from .tokenizer import MyTokenizer
from .pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Path to model directory")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the interface on")
    args = parser.parse_args()
    return args

def load_yaml(dir):
    path_to_yaml = os.path.join(dir, "model_meta.yaml")
    if not os.path.exists(path_to_yaml):
        raise Exception(f"Can't Find Model Metadata at {path_to_yaml}!!")
    with open(path_to_yaml) as f:
        meta = yaml.safe_load(f)
    model_config = meta["model_config"]
    return model_config, meta["tokenizer_path"]

def load_pipeline(directory, model_config, path_to_tokenizer):
    config = GPTConfig(**model_config)
    model = GPT(config)
    path_to_weights = os.path.join(directory, "nanochat_sft", "final_checkpoint", "model.safetensors")
    weights = mytorch.load(path_to_weights)
    model.load_state_dict(weights)
    tokenizer = MyTokenizer(path_to_tokenizer)
    pipe = Pipeline(model, tokenizer)
    return pipe

def extract_content(content):
    """Handle both string content and Gradio's list-of-dicts format"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)
    return ""

def create_chat_function(pipeline):
    def chat(message, history, temperature, topk, repetition_penalty):
        messages = []
        for entry in history:
            if isinstance(entry, dict):
                content = extract_content(entry.get("content", ""))
                if content:
                    messages.append({"role": entry["role"], "content": content})
            else:
                user_msg, bot_msg = entry
                messages.append({"role": "user", "content": str(user_msg)})
                if bot_msg is not None:
                    messages.append({"role": "assistant", "content": str(bot_msg)})

        messages.append({"role": "user", "content": message})

        print(f"\n⚙️  temperature={temperature}, topk={int(topk)}, repetition_penalty={repetition_penalty}")

        generator = pipeline.generate_for_chat(
            message={"messages": messages},
            max_token_gens=512,
            temperature=temperature,
            topk=int(topk),
            repetition_penalty=repetition_penalty,
        )

        response = ""
        for token in generator:
            if isinstance(token, dict):
                continue
            response += token
            yield response

    return chat

if __name__ == "__main__":
    args = parse_args()

    print(f"Loading model from {args.directory}...")
    meta, tokenizer_path = load_yaml(args.directory)
    pipeline = load_pipeline(args.directory, meta, tokenizer_path)
    print("Model loaded successfully!")

    print("Launching Gradio interface...")
    chat_fn = create_chat_function(pipeline)

    def vote(data: gr.LikeData):
        if data.liked:
            print("Upvoted: " + str(data.value))
        else:
            print("Downvoted: " + str(data.value))

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(
            placeholder="Chat with me!",
            height=500
        )
        chatbot.like(vote, None, None)

        with gr.Accordion("⚙️ Generation Settings", open=False):
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.01, maximum=2.0, value=0.8, step=0.05,
                    label="Temperature",
                    info="Higher = more creative/random, lower = more focused"
                )
                topk = gr.Slider(
                    minimum=1, maximum=200, value=50, step=1,
                    label="Top-K",
                    info="Sample from the top K most likely tokens"
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.2, step=0.05,
                    label="Repetition Penalty",
                    info="Higher = less repetition. 1.0 = disabled"
                )

        gr.ChatInterface(
            fn=chat_fn,
            chatbot=chatbot,
            additional_inputs=[temperature, topk, repetition_penalty],
        )

    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )