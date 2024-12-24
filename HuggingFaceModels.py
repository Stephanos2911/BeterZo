import torch
from transformers import pipeline
from huggingface_hub import login
import json
import os

# Define folder paths
root_folder = os.getcwd()
results_folder = os.path.join(root_folder, "Resultaten")
conversations_folder = os.path.join(root_folder, "Conversaties")
os.makedirs(results_folder, exist_ok=True)
os.makedirs(conversations_folder, exist_ok=True)

login("geheim")

# Replace the MPS configuration with proper device detection
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_num_threads(6)  # Limit threads for M3 efficiency

# Load the Llama 3.2 3B Instruct model using the pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
            do_sample=True,  # Changed to True to avoid warning
            temperature=0.1,  # Added low temperature for more focused outputs
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

# Define the input conversation in Dutch
conversation = """
Goedemiddag, hoe gaat het vandaag met u?
Hallo… het gaat wel, denk ik.
Ik heb eigenlijk weinig energie en eerlijk gezegd ook geen motivatie om nog veel te doen.
Dat begrijp ik.
Het is heel normaal dat de behandelingen zwaar kunnen zijn en dat het moeilijk is om gemotiveerd te blijven.
Hoe ervaart u dat zelf, op dit moment?
Het is moeilijk te omschrijven.
Ik merk gewoon dat ik geen zin heb om beter te worden. 
Het voelt allemaal zo… uitzichtloos.
Ik ben eigenlijk helemaal niet bezig met herstel.
Dat klinkt inderdaad heel zwaar.
Dus u voelt zich niet zo gemotiveerd om stappen richting herstel te maken?
Nee, niet echt.
Het voelt alsof ik geen interesse in verbetering heb.
En eigenlijk ook weinig interesse in gezondheid.
Het lijkt allemaal zo ver weg, en ik ben er een beetje onverschillig over geworden.
Dat is begrijpelijk.
Soms kan het door de fysieke vermoeidheid ook lastiger worden om u gemotiveerd te voelen.
Hoe gaat het op fysiek vlak, als ik vragen mag?
Ik ben eigenlijk niet meer actief.
En als ik iets probeer, dan merk ik dat ik snel moe word.
Zelfs simpele dingen zoals opstaan of rondlopen, het kost me allemaal zoveel moeite.
Mijn uithoudingsvermogen is gewoon heel laag, en ik voel me fysiek minder sterk dan voorheen.
Dus u ervaart dat de activiteiten die u eerder deed nu moeilijker zijn?
Ja, zelfs kleine dingen zijn al een uitdaging.
Ik kan niet in staat om veel te doen en ik merk dat er echt beperkingen in beweging zijn.
Sporten of andere inspannende activiteiten zijn bijna onmogelijk geworden.
Dat klinkt echt vermoeiend.
Het lijkt alsof de behandelingen en alles daaromheen echt een enorme tol eisen.
Hoe gaat het op sociaal vlak?
Heeft u mensen in uw omgeving met wie u hierover kunt praten?
Eigenlijk niet… ik voel me vaak alleen.
Ik heb geen sociale contacten meer om mee te praten en voel me eigenlijk best wel afgesloten van anderen.
Dat moet erg zwaar voor u zijn.
Dus u ervaart weinig steun vanuit uw omgeving?
Ja, het voelt alsof ik geen steun van vrienden heb.
Er zijn nauwelijks mensen met wie ik nog praat, en ik merk dat ik steeds minder deel uitmaak van sociale activiteiten.
Mijn sociale kring is echt heel beperkt geworden, en ik voel me vaak buitengesloten.
Dat moet een ontzettend moeilijk gevoel zijn.
U hebt het gevoel dat u er alleen voor staat?
Ja, precies.
Die sociale eenzaamheid is soms echt overweldigend.
Ik voel een gebrek aan interactie, en het maakt alles gewoon nog moeilijker.
Het lijkt soms alsof niemand echt begrijpt wat ik doormaak.
Dat is heel begrijpelijk.
Het kan soms ook lastig zijn voor anderen om de situatie volledig te bevatten, zeker wanneer ze niet weten wat u doormaakt.
Denkt u dat er iets is wat we samen kunnen doen om u een beetje meer verbonden te laten voelen?
Ik weet het niet… ik voel me gewoon zo onverschillig over alles.
Ik heb het gevoel dat ik weinig energie overhoud om te proberen.
Misschien helpt het wel, maar ik zie niet hoe ik dat moet aanpakken.
Dat begrijp ik goed.
Misschien kunnen we beginnen met kleine stappen, zonder dat het meteen te veel druk op u legt.
Zou het u bijvoorbeeld helpen om een keer deel te nemen aan een ondersteuningsgroep, of misschien een afspraak te maken met iemand die ervaring heeft met deze situaties?
Dat zou wel een optie kunnen zijn… al weet ik niet of ik daar de kracht voor heb.
Maar ik wil het wel proberen, denk ik.
Dat is al een mooie stap.
Het hoeft niet veel energie te kosten, maar misschien kan zo’n kleine stap u helpen om het gevoel van eenzaamheid een beetje te verlichten en wat sociale steun te ervaren.
En als het toch te zwaar voelt, kunnen we samen naar andere opties kijken.
Ja… dat klinkt wel goed.
Misschien is het inderdaad het proberen waard.
Al blijft het moeilijk.
Dat begrijp ik.
We kunnen rustig aan doen en stapje voor stapje verder kijken.
Het belangrijkste is dat u weet dat er mensen zijn die er voor u willen zijn.
We zullen er alles aan doen om u zo goed mogelijk te ondersteunen.
"""

system_instructions = """
You are a medical assistant that only outputs JSON. You reply in JSON format with fields representing NANDA domains.

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

You will be given a conversation between a doctor and a cancer patient. Extract only the patient's sentences and classify each one into the most relevant NANDA domain. Before classifying a patient's response, double-check if the doctor’s question aligns with a specific NANDA domain to guide your classification. Ensure that each sentence is assigned to only one domain by selecting the best-fitting option, even if multiple domains could be relevant.

In addition to classifying the patient's responses, generate a general advice based on the analysis, focusing specifically on the psychosocial aspects of the patient's experience. The advice must consist of short, actionable steps the doctor can take to address the psychosocial challenges identified in the patient's responses.

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
    "Advice": "Verwijs de patiënt naar een psycholoog voor stress- en emotionele ondersteuning. Moedig deelname aan sociale ondersteuningsgroepen aan om isolatie te verminderen. Adviseer een geleidelijk plan voor fysieke activiteiten om energieniveaus te verbeteren. Behandel fysieke symptomen zoals hoofdpijn en spierpijn met passende interventies."
}

"""

# Define models to test
models_to_test = [
    # "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Process each model
for model_id in models_to_test:
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

