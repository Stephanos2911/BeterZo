import torch
from transformers import pipeline
from huggingface_hub import login
import json, os, whisper

# Configuration
UseWhisper = True  # Set to False to disable Whisper STT
audio_file_name = "conversatie_1.mp3"  # Name of the audio file to transcribe

# Define folder paths
root_folder = os.getcwd()
results_folder = os.path.join(root_folder, "Resultaten")
conversations_folder = os.path.join(root_folder, "Conversaties", "recordings")
transcriptions_folder = os.path.join(root_folder, "Conversaties", "transcriptions")
os.makedirs(results_folder, exist_ok=True)
os.makedirs(conversations_folder, exist_ok=True)
os.makedirs(transcriptions_folder, exist_ok=True)


# Hugging Face login
login("geheim")

# Replace the MPS configuration with proper device detection
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load Whisper model
def transcribe_audio(file_path):
    model = whisper.load_model("base") 
    result = model.transcribe(file_path, fp16=torch.cuda.is_available())
    return result["text"]

# Determine conversation content
def load_conversation(use_whisper, audio_file_name, text_file_name):
    if use_whisper:
        audio_file_path = os.path.join(conversations_folder, audio_file_name)
        if os.path.exists(audio_file_path):
            try:
                print(f"Transcribing audio file: {audio_file_name}")
                return transcribe_audio(audio_file_path)
            except Exception as e:
                print(f"Error during transcription: {e}")
                return ""
        else:
            print(f"Audio file not found: {audio_file_path}")
            return ""
    else:
        text_file_path = os.path.join(transcriptions_folder, text_file_name)
        if os.path.exists(text_file_path):
            try:
                print(f"Reading text file: {text_file_name}")
                with open(text_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading text file: {e}")
                return ""
        else:
            print(f"Text file not found: {text_file_path}")
            return ""



def process_model(model_id, conversation, system_instructions):
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map=device,
            torch_dtype=torch.float16,
            model_kwargs={"low_cpu_mem_usage": True}
        )

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": conversation}
        ]

        outputs = pipe(
            messages,
            max_new_tokens=512,
            num_return_sequences=1,
            pad_token_id=pipe.tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.1,
        )

        # Handle different output formats
        if isinstance(outputs, list):
            response_text = outputs[0]['generated_text'] if isinstance(outputs[0], dict) else outputs[0]
        else:
            response_text = outputs['generated_text'] if isinstance(outputs, dict) else str(outputs)
        
        # Try to find the last valid JSON in the response
        try:
            last_opening_brace = response_text.rfind('{')
            last_closing_brace = response_text.rfind('}')
            
            if last_opening_brace != -1 and last_closing_brace != -1 and last_opening_brace < last_closing_brace:
                json_text = response_text[last_opening_brace:last_closing_brace + 1]
                response_json = json.loads(json_text)
            else:
                raise json.JSONDecodeError("No valid JSON found", response_text, 0)
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, save the raw response
            response_json = {
                "error": "Invalid JSON",
                "raw_response": response_text
            }
        
        return response_json

    except Exception as e:
        return {
            "error": str(e),
            "raw_response": "Model processing failed"
        }

#Prompt
system_instructions = """
You are a medical assistant that only outputs JSON. You reply in JSON format with fields representing NANDA domains.
You will be given a conversation between a doctor and a cancer patient. Extract only the patient's sentences and classify each one into the most relevant NANDA domain. 

You are only allowed to use the following domains: 
- Gezondheidsbevordering: Patiëntbeschrijvingen die betrekking hebben op het bevorderen van gezondheid, zoals inspanningen om gezond te blijven.
- Voeding: Alles wat betrekking heeft op voedselinname, eetgewoontes en de voedingsstatus van de patiënt.
- Eliminatie en Uitwisseling: Gaat over uitscheiding van afvalstoffen, zoals plassen, ontlasting, en ademhaling.
- Activiteit/Rust: Verwijst naar energie, lichamelijke activiteiten, vermoeidheid, slaap, en rust.
- Waarneming/Cognitie: Heeft te maken met denken, geheugen, concentratie, en de waarneming van de wereld om hen heen.
- Zelfperceptie: Betreft de manier waarop de patiënt zichzelf ziet, inclusief eigenwaarde, gevoelens van zelfvertrouwen of onzekerheid.
- Rolrelaties: Betrokken bij sociale interacties en de rol van de patiënt in familie of werk.
- Seksualiteit: Verwijst naar intieme en seksuele gevoelens of relaties.
- Omgaan met Stress/Stress Tolerantie: Verwijst naar hoe de patiënt omgaat met stress, angst en spanning.
- Levensprincipes: Betreft fundamentele waarden, overtuigingen en de ethische principes die het leven van de patiënt sturen.
- Veiligheid/Bescherming: Heeft betrekking op gevoelens van veiligheid, bescherming, en het vermijden van risico's.
- Comfort: Verwijst naar lichamelijk en geestelijk comfort, zoals pijn of ongemak.
- Groei/Ontwikkeling: Alles wat te maken heeft met de fysieke, emotionele en mentale ontwikkeling van de patiënt.

Before classifying a patient's response, double-check if the doctor’s question aligns with a specific NANDA domain to guide your classification. 
Ensure that each sentence is assigned to only one domain by selecting the best-fitting option, even if multiple domains could be relevant.

In addition to classifying the patient's responses, generate a general advice based on the analysis, focusing specifically on the psychosocial aspects of the patient's experience. 
The advice must consist of short, actionable steps the doctor can take to address the psychosocial challenges identified in the patient's responses.

Here is an example conversation and response:

Hoe voelt u zich de laatste tijd?
Ik voel me de hele tijd uitgeput.

Heeft dit invloed gehad op uw werk of sociale leven?
Ja, ik vind het moeilijk om me op werk te concentreren en ik heb sociale activiteiten vermeden.

Heeft u naast de vermoeidheid ook fysieke klachten opgemerkt?
Ja, ik heb constant hoofdpijn en mijn spieren doen de hele tijd pijn.

Zorgt u goed voor uzelf, bijvoorbeeld door goed te eten en te slapen?
Ik sla maaltijden over en vind het moeilijk om ’s nachts te slapen.

Example response (JSON):
{
    "NandaDomains": {
        "Gezondheidsbevordering": [],
        "Voeding": [
            "Ik sla maaltijden over."
        ],
        "Eliminatie en Uitwisseling": [],
        "Activiteit/Rust": [
            "Ik voel me de hele tijd uitgeput.",
            "Ik vind het moeilijk om 's nachts te slapen."
        ],
        "Waarneming/Cognitie": [
            "Ik vind het moeilijk om me op werk te concentreren."
        ],
        "Zelfperceptie": [],
        "Rolrelaties": [
            "Ik heb sociale activiteiten vermeden."
        ],
        "Seksualiteit": [],
        "Omgaan met Stress/Stress Tolerantie": [],
        "Levensprincipes": [],
        "Veiligheid/Bescherming": [],
        "Comfort": [
            "Ik heb constant hoofdpijn.",
            "Mijn spieren doen de hele tijd pijn."
        ],
        "Groei/Ontwikkeling": []
    },
    "Advice": "Verwijs de patiënt naar een psycholoog voor stress- en emotionele ondersteuning. 
    Moedig deelname aan sociale ondersteuningsgroepen aan om isolatie te verminderen. 
    Adviseer een geleidelijk plan voor fysieke activiteiten om energieniveaus te verbeteren. 
    Behandel fysieke symptomen zoals hoofdpijn en spierpijn met passende interventies."
}

"""

# Define models to test
models_to_test = [
    # "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Process each model
for model_id in models_to_test:
    UseWhisper = True  # Set to False If you don't want to use Whisper for STT but use a text file instead
    audio_file_name = "conversatie_1.mp3"  # Name of the audio file to transcribe (/conversaties/recordings/filename.mp3)
    text_file_name = "conversatie_1.txt"  # Name of the text file if not using Whisper (conversaties/transcripties/filename.txt)
    
    # Load conversation
    conversation = load_conversation(UseWhisper, audio_file_name, text_file_name)
    
    print(f"\nProcessing model: {model_id}")
    
    response_json = process_model(model_id, conversation, system_instructions)
    
    # Generate filename with model name
    model_name = model_id.split('/')[-1]
    existing_files = sorted([f for f in os.listdir(results_folder) if f.startswith(f"Result_{model_name}")])
    next_result_id = len(existing_files) + 1
    result_file_path = os.path.join(results_folder, f"Result_{model_name}_{next_result_id}.json")

    # Save results
    result_data = {
        "Prompt": system_instructions,
        "Model Name": model_id,
        "Conversation": conversation,
        "Result": response_json
    }
    
    with open(result_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)

    print(f"Analysis saved to {result_file_path}")

