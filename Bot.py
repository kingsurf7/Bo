import logging
import requests
import whisper
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message
from ultralytics import YOLO
import asyncio
import tempfile
import os
import random

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Token Telegram - À remplacer par votre vrai token
TOKEN = "TON_TOKEN_TELEGRAM"

# Vérification du token
if not TOKEN or TOKEN == "7635358951:AAE_yNMXcLiIKyJIbf-My3v4-PHs3pcUheI":
    raise ValueError("Oups ! Il semble que le token du bot n'est pas configuré. 😅")

# Initialisation du bot
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Liste des chaînes à vérifier
CHANNELS = [
    "@Brainless_dev", 
    "@LaboratoireDuFreeSurf", 
    "@hat_tunnel", 
    "@premium_apk_made_for_you"
]

# Messages plus naturels
GREETINGS = [
    "Salut ! Comment puis-je t'aider aujourd'hui ? 😊",
    "Bonjour ! Prêt à explorer ensemble ? 🌟",
    "Coucou ! Qu'est-ce qui te amène ici aujourd'hui ? 🤗"
]

ERROR_MESSAGES = {
    "api": "Oh non ! J'ai du mal à me connecter à mon cerveau numérique... Peux-tu réessayer ? 🤔",
    "generic": "Oups ! Quelque chose s'est mal passé de mon côté. Je vais me secouer les circuits et tu peux réessayer ! 😅",
    "subscription": "Je vois que tu n'es pas encore abonné à tous nos canaux. Rejoins-nous pour débloquer toutes les fonctionnalités ! 🚀"
}

# Chargement des modèles (une seule fois au démarrage)
try:
    vision_model = YOLO("yolov8n.pt")
    speech_model = whisper.load_model("base")
    logger.info("Modèles chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles : {e}")
    raise

async def is_user_subscribed(user_id: int) -> bool:
    """Vérifie si l'utilisateur est abonné à tous les canaux requis."""
    for channel in CHANNELS:
        try:
            chat_member = await bot.get_chat_member(channel, user_id)
            if chat_member.status not in ["member", "administrator", "creator"]:
                return False
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'abonnement : {e}")
            return False
    return True

@dp.message(Command("start"))
async def start(message: types.Message):
    """Gère la commande /start."""
    user_id = message.from_user.id
    first_name = message.from_user.first_name
    try:
        if not await is_user_subscribed(user_id):
            channels_list = "\n".join(f"👉 {channel}" for channel in CHANNELS)
            await message.reply(
                f"👋 Salut {first_name} ! Pour utiliser toutes les fonctionnalités de ce bot, "
                f"tu dois être abonné à nos chaînes :\n\n{channels_list}\n\n"
                "Une fois abonné, envoie-moi à nouveau /start et on pourra commencer l'aventure ! 💫"
            )
        else:
            greeting = random.choice(GREETINGS)
            await message.reply(
                f"{greeting}\n\nJe suis BRAINLESS, ton compagnon numérique. Voici ce que je peux faire :\n\n"
                "📝 Envoie-moi du texte pour discuter\n"
                "📸 Envoie une photo pour que je l'analyse\n"
                "🎤 Envoie un message vocal que je transcrirai\n\n"
                "Alors, on commence par quoi ? 😉"
            )
    except Exception as e:
        logger.error(f"Erreur dans /start : {e}")
        await message.reply("Oups ! J'ai eu un petit bug... Peux-tu réessayer ? 🐞")

@dp.message()
async def handle_text(message: Message):
    """Gère les messages textuels."""
    try:
        prompt = message.text.strip()
        if not prompt:
            await message.reply("Hmm... Je n'ai rien reçu. Peux-tu répéter ? 🧐")
            return

        # Simulation de "réflexion"
        await message.reply_chat_action("typing")
        await asyncio.sleep(1)
        
        encoded_prompt = requests.utils.quote(prompt)
        api_url = f"https://bk9.fun/ai/blackbox?q={encoded_prompt}"
        
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if not data.get("BK9"):
            await message.reply("Je suis désolé, je n'ai pas trouvé de réponse pertinente... Peux-tu reformuler ? 🤔")
            return
            
        reply_text = data["BK9"]
        
        # Ajout d'une touche personnelle
        reactions = ["Voici ce que j'en pense :", "J'ai creusé la question :", "Après réflexion :", "Voici ma réponse :"]
        chosen_reaction = random.choice(reactions)
        
        await message.reply(
            f"✨ *{chosen_reaction}*\n\n{reply_text}\n\n---\n_Tu veux explorer autre chose ?_ 😊", 
            parse_mode="Markdown"
        )
    except requests.RequestException as e:
        logger.error(f"Erreur API : {e}")
        await message.reply(ERROR_MESSAGES["api"])
    except Exception as e:
        logger.error(f"Erreur dans handle_text : {e}")
        await message.reply(ERROR_MESSAGES["generic"])

@dp.message(F.photo)
async def handle_image(message: types.Message):
    """Gère les images envoyées."""
    try:
        # Message pendant le traitement
        processing_msg = await message.reply("🔍 Je scrute ton image... Donne-moi une seconde !")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # Téléchargement de la photo
            await message.photo[-1].download(destination_file=temp_file.name)
            
            # Analyse avec YOLO
            results = vision_model(temp_file.name)
            detections = results[0].boxes.data.tolist() if results else []
            
            if not detections:
                await processing_msg.edit_text("Je n'ai rien détecté sur cette image... Es-tu sûr qu'elle n'est pas vide ? 😅")
            else:
                # Formatage plus naturel des résultats
                objects = {}
                for det in results[0].boxes.data:
                    name = det['name']
                    conf = det['confidence']*100
                    if name in objects:
                        objects[name] = max(objects[name], conf)
                    else:
                        objects[name] = conf
                
                # Tri par confiance
                sorted_objects = sorted(objects.items(), key=lambda x: x[1], reverse=True)
                
                # Création du message
                if len(sorted_objects) == 1:
                    obj, conf = sorted_objects[0]
                    response = f"Je vois un {obj} avec {conf:.1f}% de confiance ! 👀"
                else:
                    response = "Voici ce que j'ai repéré :\n"
                    for obj, conf in sorted_objects:
                        response += f"- {obj} ({conf:.1f}% de confiance)\n"
                    response += "\nC'est bien ça ? 😊"
                
                await processing_msg.edit_text(response)
                
    except Exception as e:
        logger.error(f"Erreur dans handle_image : {e}")
        await message.reply("Oh là là ! Mon analyse d'image a bugué... Peux-tu essayer avec une autre photo ? 📸")
    finally:
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@dp.message(F.voice)
async def handle_voice(message: types.Message):
    """Gère les messages vocaux."""
    try:
        # Message pendant le traitement
        processing_msg = await message.reply("🎧 J'écoute attentivement... Un instant !")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_file:
            # Téléchargement du vocal
            await message.voice.download(destination_file=temp_file.name)
            
            # Transcription
            transcription = speech_model.transcribe(temp_file.name)
            text = transcription.get('text', '').strip()
            
            if not text:
                await processing_msg.edit_text("Je n'ai pas compris ton message vocal... Peux-tu le répéter plus clairement ? 🎤")
            else:
                # Réponse avec emoji aléatoire
                emojis = ["📝", "✍️", "🗒️", "🎤"]
                await processing_msg.edit_text(
                    f"{random.choice(emojis)} Voici ce que j'ai entendu :\n\n"
                    f"\"{text}\"\n\n"
                    "C'est bien ça ? Sinon, n'hésite pas à me le ré-envoyer ! 😊"
                )
                
    except Exception as e:
        logger.error(f"Erreur dans handle_voice : {e}")
        await message.reply("Oups ! Mon oreille numérique a des acouphènes... Peux-tu renvoyer ton message ? 🦻")
    finally:
        if 'temp_file' in
