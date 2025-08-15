from twilio.rest import Client
from .config import settings
from .logger import logger
from deep_translator import GoogleTranslator

# Initialize Twilio client
twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

# Initialize translators
hindi_translator = GoogleTranslator(source='en', target='hi')
english_translator = GoogleTranslator(source='hi', target='en')

def hinglish_converter(text: str, target_language: str = 'hindi') -> str:
    """
    Convert text between English and Hinglish (Hindi written in English).
    
    Args:
        text (str): The text to convert
        target_language (str): Target language ('hindi' or 'english')
        
    Returns:
        str: Converted text
    """
    try:
        if target_language.lower() == 'hindi':
            # Convert English to Hindi
            return hindi_translator.translate(text)
        else:
            # Convert Hindi to English
            return english_translator.translate(text)
    except Exception as e:
        logger.error(f"Error in hinglish conversion: {str(e)}", exc_info=True)
        return text  # Return original text if translation fails

async def send_whatsapp_message(to_number: str, message: str) -> None:
    """
    Send a WhatsApp message using Twilio.
    
    Args:
        to_number (str): The recipient's phone number in E.164 format
        message (str): The message to send
    """
    try:
        # Send message using Twilio
        message = twilio_client.messages.create(
            body=message,
            from_=f'whatsapp:{settings.TWILIO_NUMBER}',
            to=f'whatsapp:{to_number}'
        )
        logger.info(f"WhatsApp message sent successfully - SID: {message.sid}")
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {str(e)}", exc_info=True)
        raise 