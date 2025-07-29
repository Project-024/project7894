import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ------------------------------
# Emotion Tone Rules
# ------------------------------
emotion_tone_instructions = {
    "Angry": "Let's break this down clearly and calmly. ",
    "Happy": "Let's explore this exciting topic together! ",
    "Sad": "We'll take it slowly with a simple explanation. ",
    "Fearful": "No rush, we'll go through this step-by-step. ",
    "Neutral": "No change",
    "Surprised": "This sounds exciting! Let's dive into it! ",
   
}

# ------------------------------
# Load Trained Emotion Model (.pt)
# ------------------------------
def load_emotion_model(model_path="emotion_lstm_model.pt", num_classes=7):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ------------------------------
# Predict Emotion from Image
# ------------------------------
def predict_emotion(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()

    class_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    return class_names[predicted_idx]

# ------------------------------
# Modify User Query Based on Emotion
# ------------------------------
def modify_query_with_emotion(emotion, query):
    tone = emotion_tone_instructions.get(emotion, "")
    return tone + query

# ------------------------------
# Generate Explanation with GPT-2
# ------------------------------
def generate_explanation(modified_prompt, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    inputs = tokenizer.encode(modified_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response[len(modified_prompt):].strip()

# ------------------------------
# Complete Pipeline Function
# ------------------------------
def emotion_aware_explanation(image_path, user_query):
    model_path = "emotion_lstm_model.pt"
    if not os.path.exists(model_path):
        print(" Model not found. Train the model and save it as 'emotion_lstm_model.pt'.")
        return

    # Load Model & Predict Emotion
    model = load_emotion_model(model_path)
    emotion = predict_emotion(model, image_path)
    tone_instruction = emotion_tone_instructions.get(emotion, "")

    # Modify Prompt and Generate Response
    modified_prompt = modify_query_with_emotion(emotion, user_query)
    explanation = generate_explanation(modified_prompt)

    # Display Image and Emotion + Response
    try:
        img = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        full_text = (
            f" Emotion: {emotion}\n"
            f" Tone: {tone_instruction.strip()}\n\n"
            f" GPT-2 Response:\n{explanation}"
        )
        plt.figtext(0.5, -0.05, full_text, wrap=True, ha="center", fontsize=12)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f" Could not display image: {e}")

    # Console Summary
    print("\n--- Workflow Summary ---")
    print(f" User Query: {user_query}")
    print(f" Detected Emotion: {emotion}")
    print(f" Applied Rule: {tone_instruction}")
    print(f" GPT-2 Output: {explanation}")


# ------------------------------
# Run Example
# ------------------------------
if __name__ == "__main__":
    # Example usage:
    image_path = "example_emotion.jpg"  # <-- replace with your test image
    user_query = "Write Python code for calculating the factorial of a number."
    emotion_aware_explanation(image_path, user_query)
