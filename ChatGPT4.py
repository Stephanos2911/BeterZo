from openai import OpenAI

client = OpenAI(api_key="nooit API keys naar github pushen")
import json, os

# Set your API key here
results_folder = "Resultaten"  # Folder to save JSON results
os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists
model_id = "gpt-4o"

# Define the system instructions
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

# Define the conversation
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

# Call the OpenAI API
response = client.chat.completions.create(model="gpt-4o",
messages=[
    {"role": "system", "content": system_instructions},
    {"role": "user", "content": conversation},
],
temperature=0.7)

# Parse the response content and clean the JSON
raw_response = response.choices[0].message.content
cleaned_response = raw_response.strip("```json").strip("```").strip()

# Convert the cleaned string to a JSON object
try:
    response_json = json.loads(cleaned_response)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    response_json = {"error": "Invalid JSON format in API response", "raw": cleaned_response}

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