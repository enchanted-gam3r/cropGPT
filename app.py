"""
CropGPT - AI Farming Assistant Chatbot UI
Beautiful gradient interface with custom styling
"""

import gradio as gr
from unsloth import FastLanguageModel
import torch

MODEL_LOADED = False
model = None
tokenizer = None

def load_model():
    """Load the fine-tuned farming chatbot model"""
    global model, tokenizer, MODEL_LOADED

    print("Loading model...")

    try:
        # Load base model with LoRA adapters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="farming_chatbot_lora",  # Your saved LoRA path
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)
        MODEL_LOADED = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model not loaded: {e}")
        print("Running in demo mode...")
        MODEL_LOADED = False

# Try to load model on startup
load_model()


def chat_response(message, history):
    """Generate response from the farming chatbot"""

    if not message or message.strip() == "":
        return history

    # Demo responses when model is not loaded
    if not MODEL_LOADED:
        demo_responses = {
            "punjab": "Based on Punjab's climate, I recommend:\n\n1. **Rice (Kharif season)** - Ideal for humid conditions\n2. **Wheat (Rabi season)** - Main winter crop\n3. **Cotton** - Grows well in hot weather\n4. **Sugarcane** - Thrives in humid conditions\n5. **Maize** - Suitable for summer\n\nPunjab's fertile soil and irrigation facilities make these crops highly profitable.",

            "weather": "For current weather conditions:\n\n🌞 **Sunny weather**: Great for wheat, cotton, and vegetables\n🌧️ **Rainy season**: Perfect for rice cultivation\n❄️ **Winter**: Ideal for wheat and mustard\n\nAlways check local weather forecasts before planting!",

            "pest": "To protect crops from pests:\n\n1. Use **integrated pest management (IPM)**\n2. Rotate crops regularly\n3. Maintain field hygiene\n4. Use neem-based organic pesticides\n5. Monitor fields weekly for early detection\n6. Encourage natural predators like ladybugs",

            "fertilizer": "For healthy crop growth:\n\n🌱 **Nitrogen (N)**: Promotes leaf growth\n🌾 **Phosphorus (P)**: Supports root development\n🍅 **Potassium (K)**: Improves fruit quality\n\nUse NPK ratio based on your crop type. Organic compost is also excellent!",

            "water": "Watering guidelines:\n\n💧 Most vegetables need 1-2 inches per week\n💧 Water early morning or evening\n💧 Deep watering is better than frequent shallow watering\n💧 Check soil moisture before watering\n💧 Use drip irrigation for efficiency",
        }

        # Find matching response
        msg_lower = message.lower()
        response = None

        for key, value in demo_responses.items():
            if key in msg_lower:
                response = value
                break

        if not response:
            response = "I'm your AI farming assistant! 🌾\n\nAsk me about:\n- Crop recommendations for your region\n- Weather-based farming advice\n- Pest management solutions\n- Fertilizer guidance\n- Watering schedules\n- Growing tips\n\nWhat would you like to know?"

        # Append to history
        history.append((message, response))
        return history

    # Real model inference
    try:
        prompt = f"""Below is an instruction that describes a farming-related task. Write a response that appropriately completes the request.

### Instruction:
{message}

### Response:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()

        # Append to history
        history.append((message, response))
        return history

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}\n\nPlease try again or rephrase your question."
        history.append((message, error_msg))
        return history

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
custom_css = """
/* Main container gradient background */
.gradio-container {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%) !important;
}

/* Header styling with green gradient */
.header-container {
    background: linear-gradient(135deg, #2e7d32 0%, #388e3c 50%, #43a047 100%);
    padding: 30px;
    border-radius: 15px 15px 0 0;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 0;
}

.header-title {
    color: white;
    font-size: 48px;
    font-weight: bold;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
}

.header-subtitle {
    color: #e8f5e9;
    font-size: 20px;
    margin-top: 10px;
    font-weight: 300;
}

/* Leaf emoji as icon */
.leaf-icon {
    font-size: 56px;
    filter: drop-shadow(2px 2px 3px rgba(0,0,0,0.3));
}

/* Chat container */
#chatbot-container {
    background: white;
    border-radius: 0 0 15px 15px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Input box styling */
#msg-input textarea {
    border: 2px solid #81c784 !important;
    border-radius: 25px !important;
    padding: 15px 20px !important;
    font-size: 16px !important;
}

#msg-input textarea:focus {
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1) !important;
}

/* Send button */
.send-button {
    background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%) !important;
    border: none !important;
    border-radius: 50% !important;
    min-width: 50px !important;
    height: 50px !important;
    color: white !important;
    font-size: 20px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    transition: all 0.3s ease !important;
}

.send-button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2) !important;
}

/* Clear button */
.clear-button {
    background: linear-gradient(135deg, #ef5350 0%, #e57373 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 10px 25px !important;
    font-weight: 600 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #2e7d32;
    padding: 20px;
    font-size: 14px;
    margin-top: 20px;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #e8f5e9;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
}
"""

# GRADIO INTERFACE
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="CropGPT - AI Farming Assistant") as demo:

    # Header with gradient and leaf icon
    gr.HTML("""
        <div class="header-container">
            <div class="header-title">
                <span class="leaf-icon">🌾</span>
                <span>CropGPT</span>
            </div>
            <div class="header-subtitle">Your AI farming assistant</div>
        </div>
    """)

    # Main chat interface
    with gr.Column(elem_id="chatbot-container"):
        chatbot = gr.Chatbot(
            value=[],
            label="",
            height=450,
            show_label=False,
            avatar_images=(None, "🌾"),
            bubble_full_width=False,
            show_copy_button=True
        )

        with gr.Row():
            with gr.Column(scale=9):
                msg = gr.Textbox(
                    placeholder="Describe your region and weather conditions...",
                    show_label=False,
                    container=False,
                    elem_id="msg-input"
                )
            with gr.Column(scale=1, min_width=50):
                submit_btn = gr.Button("➤", elem_classes="send-button")


        # Clear button
        clear = gr.Button("🗑️ Clear Chat", elem_classes="clear-button")

    # Footer
    gr.HTML("""
        <div class="footer">
            <p><strong>CropGPT</strong> - Powered by AI | Helping farmers make informed decisions 🌱</p>
            <p style="font-size: 12px; color: #66bb6a;">Ask about crop recommendations, weather forecasts, pest control, or growing tips</p>
        </div>
    """)

    # Event handlers
    msg.submit(chat_response, [msg, chatbot], [chatbot]).then(
        lambda: gr.Textbox(value=""), None, [msg]
    )

    submit_btn.click(chat_response, [msg, chatbot], [chatbot]).then(
        lambda: gr.Textbox(value=""), None, [msg]
    )

    clear.click(lambda: [], None, [chatbot])

if __name__ == "__main__":
    print("🌾 Starting CropGPT Chatbot...")
    print("=" * 70)

    if MODEL_LOADED:
        print("✅ Model loaded successfully - Using fine-tuned model")
    else:
        print("⚠️  Running in DEMO mode - Using sample responses")

    print("=" * 70)

    demo.launch(
        share=True,
        show_error=True,
        inbrowser=True,
    )