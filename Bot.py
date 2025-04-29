import logging
import os
import asyncio
import tempfile
from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable, Tuple

import requests
import whisper
from aiogram import Bot, Dispatcher, types, F, BaseMiddleware
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from ultralytics import YOLO

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration du token
TOKEN = os.getenv("TELEGRAM_TOKEN", "TON_TOKEN_TELEGRAM")
if not TOKEN or TOKEN == "TON_TOKEN_TELEGRAM":
    raise ValueError("Token Telegram manquant ou invalide")

# Initialisation du bot
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Liste des canaux requis
CHANNELS = [
    "@UniversDuFreeSurf",
    "@LaboratoireDuFreeSurf",
    "@hat_tunnel",
    "@premium_apk_made_for_you"
]

# Chargement des modèles
try:
    vision_model = YOLO("yolov8n.pt")
    speech_model = whisper.load_model("base")
    logger.info("Modèles chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles: {e}")
    raise

# Gestion des états
class UserState(StatesGroup):
    awaiting_response = State()

# Système de rate limiting
class RateLimiter:
    def __init__(self):
        self.user_last_request = {}
        self.cooldown = 5  # secondes entre requêtes

    async def check_rate_limit(self, user_id: int) -> Tuple[bool, float]:
        now = datetime.now()
        last_request = self.user_last_request.get(user_id)

        if last_request and (now - last_request) < timedelta(seconds=self.cooldown):
            remaining = (last_request + timedelta(seconds=self.cooldown) - now
            return False, remaining.total_seconds()

        self.user_last_request[user_id] = now
        return True, 0

rate_limiter = RateLimiter()

# Middleware anti-spam
class RateLimitMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Message, dict], Awaitable[Any]],
        event: Message,
        data: dict
    ) -> Any:
        user = data.get("event_from_user")
        if user:
            allowed, remaining = await rate_limiter.check_rate_limit(user.id)
            if not allowed:
                await event.answer(
                    f"⌛ Attendez {int(remaining)}s avant une nouvelle requête",
                    show_alert=True
                )
                return
        return await handler(event, data)

dp.message.middleware(RateLimitMiddleware())

async def is_user_subscribed(user_id: int) -> Tuple[bool, str]:
    """Vérifie les abonnements aux chaînes"""
    missing = []
    for channel in CHANNELS:
        try:
            chat_member = await bot.get_chat_member(channel, user_id)
            if chat_member.status not in ["member", "administrator", "creator"]:
                missing.append(channel)
        except Exception as e:
            logger.error(f"Erreur vérification {channel}: {e}")
            missing.append(channel)
    
    if missing:
        message = "Abonnez-vous à:\n" + "\n".join(f"- {ch}" for ch in missing)
        return False, message
    return True, ""

@dp.message(Command("start"))
async def start(message: types.Message):
    """Commande start"""
    user_id = message.from_user.id
    try:
        subscribed, msg = await is_user_subscribed(user_id)
        if not subscribed:
            await message.reply(f"{msg}\n\nPuis relancez /start")
            return

        await message.reply(
            "🤖 Bienvenue!\n"
            "Envoyez:\n"
            "- Texte pour réponse\n"
            "- Photo pour analyse\n"
            "- Vocal pour transcription"
        )
    except Exception as e:
        logger.error(f"Erreur /start: {e}")
        await message.reply("❌ Erreur, réessayez")

@dp.message(Command("help"))
async def help_cmd(message: types.Message):
    """Commande help"""
    help_text = (
        "🛠️ Aide:\n"
        "/start - Démarrer le bot\n"
        "/help - Afficher ce message\n\n"
        "Fonctions:\n"
        "- Réponse aux questions\n"
        "- Analyse d'images\n"
        "- Transcription vocale"
    )
    await message.reply(help_text)

@dp.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    """Gestion des messages texte"""
    try:
        # Vérification rate limit
        allowed, remaining = await rate_limiter.check_rate_limit(message.from_user.id)
        if not allowed:
            await message.reply(f"⏳ Attendez {int(remaining)}s")
            return

        await state.set_state(UserState.awaiting_response.state)
        await bot.send_chat_action(message.chat.id, "typing")

        # Traitement du texte
        prompt = message.text.strip()
        if not prompt:
            await message.reply("ℹ️ Message vide")
            return

        # Simulation traitement
        await asyncio.sleep(1)

        # Requête API
        response = requests.get(
            f"https://bk9.fun/ai/blackbox?q={requests.utils.quote(prompt)}",
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        reply = data.get("BK9", "Pas de réponse disponible")
        await message.reply(f"💡 Réponse:\n{reply}")

    except requests.RequestException:
        await message.reply("🌐 Erreur réseau")
    except Exception as e:
        logger.error(f"Erreur texte: {e}")
        await message.reply("❌ Erreur de traitement")
    finally:
        await state.clear()

@dp.message(F.photo)
async def handle_image(message: types.Message, state: FSMContext):
    """Gestion des images"""
    try:
        # Vérification rate limit
        allowed, remaining = await rate_limiter.check_rate_limit(message.from_user.id)
        if not allowed:
            await message.reply(f"⏳ Attendez {int(remaining)}s")
            return

        await state.set_state(UserState.awaiting_response.state)
        await bot.send_chat_action(message.chat.id, "upload_photo")

        # Téléchargement photo
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            file_path = tmp_file.name
            await message.photo[-1].download(destination=file_path)

        # Analyse image
        results = vision_model(file_path)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detections.append({
                    'name': result.names[int(box.cls)],
                    'confidence': float(box.conf)
                })

        # Réponse
        if not detections:
            await message.reply("🔍 Aucun objet détecté")
        else:
            response = "🖼️ Objets détectés:\n" + "\n".join(
                f"- {obj['name']} ({obj['confidence']*100:.1f}%)" 
                for obj in detections
            )
            await message.reply(response)

    except Exception as e:
        logger.error(f"Erreur image: {e}")
        await message.reply("❌ Erreur d'analyse")
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.unlink(file_path)
        await state.clear()

@dp.message(F.voice)
async def handle_voice(message: types.Message, state: FSMContext):
    """Gestion des messages vocaux"""
    try:
        # Vérification rate limit
        allowed, remaining = await rate_limiter.check_rate_limit(message.from_user.id)
        if not allowed:
            await message.reply(f"⏳ Attendez {int(remaining)}s")
            return

        await state.set_state(UserState.awaiting_response.state)
        await bot.send_chat_action(message.chat.id, "typing")

        # Téléchargement audio
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp_file:
            file_path = tmp_file.name
            await message.voice.download(destination=file_path)

        # Transcription
        result = speech_model.transcribe(file_path)
        text = result.get('text', '').strip()
        
        if text:
            await message.reply(f"🎤 Transcription:\n{text}")
        else:
            await message.reply("🔇 Aucune transcription disponible")

    except Exception as e:
        logger.error(f"Erreur vocal: {e}")
        await message.reply("❌ Erreur de transcription")
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.unlink(file_path)
        await state.clear()

if __name__ == "__main__":
    from aiogram import executor
    try:
        logger.info("Démarrage du bot en mode polling...")
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        logger.error(f"Erreur fatale: {e}") 
