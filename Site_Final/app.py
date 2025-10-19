from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import pandas as pd
import os

app = Flask(__name__)

# 🔑 Cheia API (din variabile de mediu)
client = OpenAI(api_key="")

# 📊 Încarcă baza de date
df = pd.read_excel("Data_Base.xlsx")

# 🔍 Cuvinte-cheie pentru detectarea rapidă a problemelor
keywords = {
    "anxietate_generalizata": "Anxietate generalizata",
    "fobie": "Fobie specifica",
    "tulburare_obsesiv_compulsiva": "Tulburare obsesiv-compulsiva",
    "ptsd": "Tulburare de stres post-traumatic",
    "tulburare_bipolara": "Tulburare bipolara",
    "schizofrenie": "Schizofrenie",
    "tulburare_alimentara": "Tulburare alimentara",
    "insomnie": "Insomnie",
    "adictie": "Adictie",
    "stress": "Stress cronic",
    "mania": "Episod maniacal",
    "hipomanie": "Hipomanie",
    "tulburare_de_personalitate": "Tulburare de personalitate",
    "agorafobie": "Agorafobie",
    "claustrofobie": "Claustrofobie",
    "depresie": "Depresie",
    "anxietate": "Anxietate sociala",
    "suicid": "Ganduri suicidare",
    "burnout": "Burnout",
    "oboseala_cronica": "Oboseala cronica",
    "oboseala": "Oboseala cronica",
}

# 🔹 Caută sfaturi similare în baza de date
def gaseste_sfaturi(problema_user, max_solutions=3):
    similar_rows = df[df['Problema_emotionala'].str.contains(problema_user, case=False, na=False)]
    if similar_rows.empty:
        return []
    return similar_rows['Solutie'].sample(min(max_solutions, len(similar_rows))).tolist()


@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    conversation_history = data.get("history", [])  # permite context conversațional

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # 1️⃣ Încearcă detectarea prin cuvinte cheie
    problema_detectata = ""
    for k, v in keywords.items():
        if k in user_message.lower():
            problema_detectata = v
            break

    # 2️⃣ Dacă nu găsește nimic, folosește GPT pentru a interpreta mesajul
    if not problema_detectata:
        try:
            interpretation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ești un psiholog virtual. Primești un mesaj și trebuie să identifici problema emoțională principală. "
                            "Alege una dintre: " + ", ".join(keywords.values())
                        ),
                    },
                    {"role": "user", "content": f"Mesaj: {user_message}"},
                ],
            )
            problema_detectata = interpretation.choices[0].message.content.strip()
        except Exception:
            problema_detectata = "Stres cronic"

    # 3️⃣ Extrage sfaturi (dacă există)
    sfaturi = gaseste_sfaturi(problema_detectata, max_solutions=3)
    sfaturi_text = "; ".join(sfaturi) if sfaturi else "Nicio soluție exactă găsită."

    # 4️⃣ Creează context conversațional complet
    chat_context = [{"role": "system", "content": (
        "Ești MindEase+, un asistent empatic de sănătate mintală. "
        "Scopul tău este să asculți, să înțelegi emoțiile utilizatorului și să oferi sprijin empatic. "
        "Poți folosi sugestii din baza de date (dacă sunt relevante), dar ai libertatea să oferi și sfaturi personalizate. "
        "Nu oferi diagnostice medicale — doar suport emoțional, claritate și motivație. "
        "Răspunsurile tale trebuie să sune naturale, ca o conversație sinceră, caldă și atentă."
    )}]

    # Adaugă istoricul conversației (dacă există)
    for msg in conversation_history:
        chat_context.append({"role": msg["role"], "content": msg["content"]})

    # Adaugă mesajul curent al utilizatorului
    chat_context.append({"role": "user", "content": user_message})

    # 5️⃣ Trimite tot contextul către GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_context + [
            {
                "role": "system",
                "content": (
                    f"Problema detectată: {problema_detectata}. "
                    f"Din baza de date există sugestiile: {sfaturi_text}. "
                    "Integrează-le doar dacă sunt potrivite contextului. "
                    "Răspunde ca un prieten înțelept, empatic și sincer, nu intra in detalii doar fi dornic sa ajuti "
                ),
            }
        ],
        temperature=0.85,
        max_tokens=400,
    )

    ai_message = response.choices[0].message.content
    return jsonify({"reply": ai_message, "detected_problem": problema_detectata})


if __name__ == "__main__":
    app.run(debug=True)
