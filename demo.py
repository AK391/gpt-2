from transformers import pipeline, set_seed
import gradio as gr

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

def gpt2(text):
    return generator(text, max_length=30, num_return_sequences=1)[0]['generated_text']


inputs = gr.inputs.Textbox(lines=1, label="Input Text")
outputs =  gr.outputs.Textbox(label="GPT-2")

title = "GPT-2"
description = "demo for OpenAI GPT-2. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://openai.com/blog/better-language-models/'>Better Language Models and Their Implications</a> | <a href='https://github.com/openai/gpt-2'>Github Repo</a></p>"
examples = [
            ["Hi"],
            ["What is OpenAI?"],
]

gr.Interface(gpt2, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()