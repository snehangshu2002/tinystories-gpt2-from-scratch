import gradio as gr
import torch
from transformers import pipeline

# 1. Load the model via Hugging Face pipeline
# We use the pipeline because it's the cleanest way to do generation
print("Loading model... this may take a moment.")
generator = pipeline(
    "text-generation",
    model="snehangshu511/tinystories-gpt2-124M-scratch",
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("Model loaded successfully!")

# 2. Define the function that Gradio will call when the user clicks 'Generate'
def generate_story(prompt, temperature, max_tokens):
    if not prompt.strip():
        return "Please enter a prompt to start the story!"
        
    try:
        # Generate the text
        output = generator(
            prompt,
            max_new_tokens=max_tokens,
            max_length=None,
            pad_token_id=generator.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_k=50
        )
        return output[0]['generated_text']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# 3. Build the minimal Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 TinyStories GPT-2 (124M)
        Built from scratch in PyTorch! Enter a starting sentence below and watch the model write a short story.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            prompt_input = gr.Textbox(
                label="Story Prompt", 
                placeholder="Once upon a time, there was a little dog named Max...",
                lines=3
            )
            
            with gr.Row():
                # Sliders to control the output
                temp_slider = gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.1, label="Temperature (Creativity)")
                token_slider = gr.Slider(minimum=20, maximum=200, value=100, step=10, label="Max New Tokens")
            
            generate_btn = gr.Button("✨ Generate Story", variant="primary")
            
        with gr.Column(scale=3):
            # Output section
            output_text = gr.Textbox(
                label="Generated Story",
                lines=10,
                interactive=False
            )
            
    # Connect the button to the function
    generate_btn.click(
        fn=generate_story,
        inputs=[prompt_input, temp_slider, token_slider],
        outputs=output_text
    )
    
    # Allow submitting by pressing Enter in the textbox
    prompt_input.submit(
        fn=generate_story,
        inputs=[prompt_input, temp_slider, token_slider],
        outputs=output_text
    )

# 4. Launch the app!
if __name__ == "__main__":
    demo.launch()
