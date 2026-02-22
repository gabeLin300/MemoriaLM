import gradio as gr


def upload_and_chat(files, url, query):
    return "Stub response. Integrate backend here."


with gr.Blocks(title="RAG NotebookLM Clone") as demo:
    gr.Markdown("# RAG NotebookLM Clone")
    files = gr.File(label="Upload documents", file_count="multiple")
    url = gr.Textbox(label="URL")
    query = gr.Textbox(label="Ask a question")
    output = gr.Textbox(label="Answer")
    submit = gr.Button("Submit")
    submit.click(upload_and_chat, inputs=[files, url, query], outputs=[output])


demo.launch()
