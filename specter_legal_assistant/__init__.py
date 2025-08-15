from fastapi import FastAPI, Request, Form, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from deep_translator import GoogleTranslator
from fastapi.responses import PlainTextResponse, StreamingResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import os
import asyncio
from typing import Optional, Dict, Any, List, Tuple
import langdetect
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from gtts import gTTS
import io
import speech_recognition as sr
import base64
from twilio.rest import Client
from fastapi.staticfiles import StaticFiles
import re
from fastapi import BackgroundTasks
from functools import lru_cache
import hashlib
import json
from cachetools import TTLCache, cached
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
from threading import Lock
from groq import Groq
import httpx
from pathlib import Path
import tempfile
import shutil

from .config import settings
from .logger import logger
from .file_manager import file_manager
from .rag_manager import rag_manager
from .utils import hinglish_converter, send_whatsapp_message

# Add reload lock
reload_lock = Lock()
last_reload_time = 0
RELOAD_COOLDOWN = 5  # seconds between reloads

# Initialize FastAPI app with rate limiting
app = FastAPI(title="Specter Legal Assistant")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add startup event handler with ready flag
app.state.ready = False
app.state.last_request_time = 0
app.state.request_count = 0

@app.middleware("http")
async def add_reload_protection(request: Request, call_next):
    """Middleware to prevent reloads during active requests."""
    global last_reload_time
    
    # Update request tracking
    app.state.last_request_time = time.time()
    app.state.request_count += 1
    
    try:
        response = await call_next(request)
        return response
    finally:
        app.state.request_count -= 1

# Initialize services
translator = GoogleTranslator(source='auto', target='en')  # For translating to English
hindi_translator = GoogleTranslator(source='en', target='hi')  # For translating back to Hindi
api_key_header = APIKeyHeader(name="X-API-Key")

# Initialize Twilio client
twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

# Initialize caches
query_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour
model_cache = TTLCache(maxsize=10, ttl=86400)   # Cache for 24 hours

# Replace the Hugging Face imports and initializations with Groq
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

# Later in the file, update the MODEL_CONFIGS
MODEL_CONFIGS = {
    "english": {
        "primary": {
            "name": "llama-3.3-70b-versatile",
            "type": "chat",
            "provider": "groq"
        }
    }
}

# Replace the hf_client initialization with Groq client
# Initialize Groq client
groq_client = Groq(api_key=settings.GROQ_API_TOKEN)

# Initialize thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Initialize model pipelines
model_pipelines = {}

# Load models asynchronously
async def load_models():
    try:
        for lang, configs in MODEL_CONFIGS.items():
            model_pipelines[lang] = {}
            
            # Try loading primary model
            try:
                primary_config = configs["primary"]
                model_name = primary_config["name"]
                
                # Create a pipeline that uses the Groq API
                model_pipelines[lang]["primary"] = {
                    "client": groq_client,
                    "model": model_name,
                    "type": primary_config["type"],
                    "provider": primary_config["provider"]
                }
                logger.info(f"Successfully loaded primary model for {lang}")
                
            except Exception as e:
                logger.error(f"Error loading primary model for {lang}: {str(e)}")
                raise
                    
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting application initialization...")
        # Check if we're in a reload cooldown period
        current_time = time.time()
        if current_time - last_reload_time < RELOAD_COOLDOWN:
            logger.warning(f"Reload cooldown active, waiting {RELOAD_COOLDOWN} seconds")
            await asyncio.sleep(RELOAD_COOLDOWN)
        await load_models()  # Re-enable model loading
        print(f"PUBLIC_BASE_URL at runtime: {settings.PUBLIC_BASE_URL}")
        app.state.ready = True
        logger.info("Application initialization complete - ready to serve requests")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        app.state.ready = False
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Handle graceful shutdown."""
    global last_reload_time
    
    # Check if there are active requests
    if app.state.request_count > 0:
        logger.warning(f"Shutdown requested with {app.state.request_count} active requests")
        # Wait for requests to complete (up to 5 seconds)
        for _ in range(50):
            if app.state.request_count == 0:
                break
            await asyncio.sleep(0.1)
    
    logger.info("Shutting down application...")
    app.state.ready = False
    last_reload_time = time.time()
    logger.info("Application shutdown complete")

def should_reload() -> bool:
    """Determine if a reload should be allowed."""
    global last_reload_time
    current_time = time.time()
    
    # Don't reload if there are active requests
    if app.state.request_count > 0:
        logger.warning(f"Reload blocked - {app.state.request_count} active requests")
        return False
    
    # Don't reload if we're in cooldown
    if current_time - last_reload_time < RELOAD_COOLDOWN:
        logger.warning(f"Reload blocked - in cooldown period")
        return False
    
    return True

@app.get("/health")
async def health_check():
    """Health check endpoint to verify application status."""
    if not app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="Application is initializing, please try again in a moment"
        )
    
    # Check model pipelines
    if not model_pipelines or "english" not in model_pipelines:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded"
        )
    
    # Check vector store
    if not rag_manager.vector_store:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not loaded"
        )
    
    return {
        "status": "healthy",
        "models_loaded": bool(model_pipelines),
        "vector_store_loaded": bool(rag_manager.vector_store),
        "ready": app.state.ready,
        "active_requests": app.state.request_count,
        "time_since_last_reload": time.time() - last_reload_time
    }

# Models
class LegalQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(..., pattern="^(english|hindi)$")

class DocumentData(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=200)
    details: str = Field(..., min_length=1, max_length=1000)

class WhatsAppMessage(BaseModel):
    body: str = Field(..., min_length=1, max_length=1000)
    sender: str = Field(..., min_length=1)
    message_type: str = Field(default="text")
    media_url: Optional[str] = None

class ConversationContext(BaseModel):
    conversation_id: str
    history: List[Dict[str, str]]
    created_at: datetime
    last_updated: datetime

# Cache for conversation history
conversation_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL

# Security
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

async def verify_twilio_request(request: Request):
    validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)
    form_data = await request.form()
    if not validator.validate(
        str(request.url),
        form_data,
        request.headers.get("X-Twilio-Signature", "")
    ):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
    return form_data

# Enhanced core functions
@cached(cache=query_cache)
def get_cached_response(query_hash: str) -> Optional[str]:
    """Get cached response for a query."""
    return query_cache.get(query_hash)

def generate_query_hash(query: str, language: str) -> str:
    """Generate a unique hash for a query."""
    return hashlib.md5(f"{query}:{language}".encode()).hexdigest()

async def get_conversation_context(conversation_id: str) -> Optional[ConversationContext]:
    """Get conversation context from cache."""
    return conversation_cache.get(conversation_id)

async def update_conversation_context(conversation_id: str, query: str, response: str):
    """Update conversation context in cache."""
    context = await get_conversation_context(conversation_id)
    if not context:
        context = ConversationContext(
            conversation_id=conversation_id,
            history=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    
    context.history.append({
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    })
    context.last_updated = datetime.now()
    conversation_cache[conversation_id] = context

async def generate_response(prompt: str, language: str = "english", **kwargs) -> str:
    try:
        pipeline = model_pipelines[language]["primary"]
        client = pipeline["client"]
        model = pipeline["model"]
        
        # Generate response using the Groq API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant that provides clear, accurate information about Indian law."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=kwargs.get("max_new_tokens", 400),
            temperature=kwargs.get("temperature", 0.7),
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise

async def ask_legal_response(query: str, language: str = "english") -> dict:
    try:
        logger.info(f"Processing query in {language}: {query}")
        
        # Validate input
        if not query or not isinstance(query, str):
            logger.error("Invalid query input")
            raise ValueError("Query must be a non-empty string")
            
        if language not in ["english", "hindi"]:
            logger.error(f"Invalid language: {language}")
            raise ValueError("Language must be either 'english' or 'hindi'")
            
        # Validate model pipeline
        if language not in model_pipelines or "primary" not in model_pipelines[language]:
            logger.error(f"Model pipeline not available for {language}")
            raise ValueError(f"Language model not available for {language}")
        
        # Enhanced domain detection with specific keywords and phrases
        domain = "consumer_rights"  # default domain
        query_lower = query.lower()
        
        # Detect emergency and personal queries
        is_emergency = False
        is_personal_query = False
        
        # Emergency indicators with more specific triggers
        emergency_indicators = {
            "criminal_law": [
                "arrested me", "police has arrested", "taken into custody",
                "detained", "in police custody", "filing fir against me",
                "charged with", "accused of", "criminal case against me"
            ],
            "family_law": [
                "dowry harassment", "domestic violence", "abuse", "harassment", "threatened",
                "forced", "coerced", "restraining order", "protection order",
                "immediate danger", "unsafe", "threat to life", "threatening me",
                "hurting me", "beating me", "mental torture", "physical abuse"
            ],
            "consumer_rights": [
                "medicine", "drug", "side effects", "allergic reaction", "health hazard",
                "dangerous product", "safety issue", "life threatening", "serious injury",
                "medical emergency", "adverse reaction", "poisoning", "toxic",
                "emergency", "urgent", "immediate action", "right now"
            ]
        }
        
        # Check for emergency situations first
        for domain_name, indicators in emergency_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                is_emergency = True
                domain = domain_name
                logger.info(f"Emergency situation detected in {domain_name}")
                break
        
        # If no emergency detected, proceed with regular domain detection
        if not is_emergency:
            # Family Law Domain (with higher priority for dowry cases)
            if "dowry" in query_lower or any(phrase in query_lower for phrase in [
                "divorce", "marriage", "custody", "maintenance", "alimony",
                "husband", "wife", "spouse", "marital", "matrimonial",
                "separation", "annulment", "restraining order", "domestic violence",
                "child custody", "visitation rights", "child support",
                "hindu marriage act", "special marriage act",
                "cruelty", "desertion", "adultery", "mental cruelty"
            ]):
                domain = "family_law"
                logger.info("Setting domain to family_law based on family law keywords")
            
            # Criminal Law Domain
            elif any(phrase in query_lower for phrase in [
                "arrest", "bail", "criminal", "police", "fir", "complaint", "offense",
                "accused", "charges", "investigation", "custody", "detention",
                "section 498a", "dowry harassment", "domestic violence", "cruelty",
                "assault", "threat", "intimidation", "harassment", "stalking"
            ]):
                domain = "criminal_law"
                logger.info("Setting domain to criminal_law based on criminal justice keywords")
            
            # Property Law Domain
            elif any(phrase in query_lower for phrase in [
                "property", "land", "house", "real estate", "ownership",
                "title", "deed", "possession", "tenant", "landlord",
                "lease", "rent", "eviction", "property tax", "encumbrance",
                "mortgage", "loan", "transfer", "sale", "purchase",
                "registration", "documentation", "inheritance", "will",
                "succession", "partition", "boundary", "dispute"
            ]):
                domain = "property_law"
                logger.info("Setting domain to property_law based on property law keywords")
            
            # Employment Law Domain
            elif any(phrase in query_lower for phrase in [
                "employment", "workplace", "harassment", "termination",
                "salary", "wages", "bonus", "leave", "working hours",
                "contract", "appointment", "resignation", "dismissal",
                "discrimination", "sexual harassment", "workplace safety",
                "industrial dispute", "labor court", "gratuity", "provident fund",
                "epf", "esi", "minimum wages", "overtime", "notice period"
            ]):
                domain = "employment_law"
                logger.info("Setting domain to employment_law based on employment law keywords")
            
            # Consumer Rights Domain
            elif any(phrase in query_lower for phrase in [
                "consumer", "defective", "product", "service", "warranty",
                "refund", "replacement", "complaint", "consumer court",
                "consumer forum", "deficiency", "unfair trade practice",
                "e-commerce", "online shopping", "delivery", "quality",
                "price", "billing", "invoice", "guarantee", "return",
                "compensation", "damages", "consumer protection act"
            ]):
                domain = "consumer_rights"
                logger.info("Setting domain to consumer_rights based on consumer law keywords")
            
            # Constitutional Law Domain
            elif any(phrase in query_lower for phrase in [
                "constitution", "fundamental rights", "article 21", "article 14",
                "article 19", "right to life", "equality", "freedom",
                "constitutional remedy", "writ", "habeas corpus",
                "mandamus", "prohibition", "certiorari", "quo warranto",
                "supreme court", "high court", "constitutional court",
                "judicial review", "public interest litigation", "pil"
            ]):
                domain = "constitutional_law"
                logger.info("Setting domain to constitutional_law based on constitutional law keywords")
            
            # Civil Law Domain
            elif any(phrase in query_lower for phrase in [
                "contract", "agreement", "breach", "damages", "compensation",
                "civil suit", "civil court", "injunction", "specific performance",
                "tort", "negligence", "defamation", "nuisance", "trespass",
                "recovery", "debt", "loan", "mortgage", "guarantee",
                "indemnity", "arbitration", "mediation", "settlement"
            ]):
                domain = "civil_law"
                logger.info("Setting domain to civil_law based on civil law keywords")
        
        # Add emergency-specific context keywords
        if is_emergency:
            emergency_context_keywords = {
                "criminal_law": "immediate legal rights arrest custody police procedure emergency",
                "family_law": "domestic violence protection order emergency shelter immediate safety",
                "consumer_rights": "product safety emergency recall immediate action health hazard"
            }
            domain_context_keywords = emergency_context_keywords.get(domain, domain_context_keywords[domain])
        
        # Add domain-specific context keywords with emergency focus
        domain_context_keywords = {
            "family_law": "dowry prohibition act section 498A IPC domestic violence act protection order immediate relief",
            "consumer_rights": "drugs and cosmetics act consumer protection act medical emergency product safety recall",
            "criminal_law": "criminal procedure code IPC section 498A dowry harassment domestic violence",
            "property_law": "transfer of property act registration title deed ownership",
            "employment_law": "industrial disputes act factories act minimum wages act",
            "constitutional_law": "constitution fundamental rights article 21 article 14",
            "civil_law": "contract act specific relief act civil procedure code"
        }

        # Define domain-specific templates for response generation
        domain_templates = {
            "criminal_law": {
                "example": "I understand you're in a difficult situation with the police. Under Indian law, you have several important rights that protect you during arrest and custody. The Constitution of India under Article 22 and the Criminal Procedure Code ensure that you have the right to be informed of the grounds of arrest, the right to consult a legal practitioner, and the right to be produced before a magistrate within 24 hours. The police must follow proper procedures during arrest, including informing your family and allowing you to make a phone call. If you believe your rights have been violated, you can file a complaint with the police station's senior officer or approach the local magistrate. It's important to remain calm and cooperate while asserting your legal rights.",
                "key_elements": ["constitutional rights", "arrest procedures", "legal representation", "time limits", "judicial safeguards"]
            },
            "family_law": {
                "example": "I understand you're going through a difficult time in your marriage. Under the Hindu Marriage Act of 1955, you have several legal options available. The law recognizes various grounds for divorce including cruelty, desertion, and adultery. For cases involving dowry harassment, you can file a complaint under Section 498A of the Indian Penal Code. The court can also grant you maintenance, child custody, and protection orders if needed. The process begins by filing a petition in the family court, and you have the right to legal aid if you cannot afford a lawyer. The court will consider all aspects of your case including evidence of harassment, financial circumstances, and the welfare of any children involved.",
                "key_elements": ["marriage laws", "divorce grounds", "maintenance", "custody", "protection orders"]
            },
            "property_law": {
                "example": "I understand you're dealing with a property-related issue. Under the Transfer of Property Act and other relevant laws, property transactions require proper documentation and registration. When purchasing property, you need to verify the title deed, check for encumbrances, and ensure proper registration of the sale deed. The seller must have clear title to the property, and all necessary permissions and approvals should be in place. If you're facing issues with property possession or documentation, you can approach the civil court for specific performance or file a suit for declaration of title. The court will examine the property documents, ownership history, and any existing disputes before making a decision.",
                "key_elements": ["property documents", "registration", "title verification", "legal remedies", "court procedures"]
            },
            "employment_law": {
                "example": "I understand you're facing issues at your workplace. Under Indian employment laws, you have several protections against workplace harassment and unfair treatment. The Sexual Harassment of Women at Workplace Act provides a framework for addressing harassment complaints, while the Industrial Disputes Act protects against unfair termination. Your employer must follow proper procedures for any disciplinary action, including giving you a chance to explain your position. If you've been terminated, you're entitled to notice period compensation and other benefits as per your employment contract. You can approach the labor court or file a complaint with the labor commissioner if you believe your rights have been violated.",
                "key_elements": ["workplace rights", "harassment laws", "termination procedures", "compensation", "legal remedies"]
            },
            "consumer_rights": {
                "example": "I understand you've received a defective product. Under the Consumer Protection Act, you have several legal remedies available. You can demand a replacement, refund, or compensation for the defective product. The law protects your right to receive goods of proper quality and service. You can file a complaint with the consumer forum, and the process is designed to be simple and quick. The forum can order the seller to replace the product, refund your money, or pay compensation for any damages or inconvenience caused. It's important to keep all receipts, warranty cards, and communication with the seller as evidence.",
                "key_elements": ["consumer rights", "product quality", "refund procedures", "compensation", "complaint filing"]
            },
            "constitutional_law": {
                "example": "I understand you're concerned about your fundamental rights. The Constitution of India guarantees several fundamental rights under Part III, including the right to equality, freedom, and protection against discrimination. Article 21 ensures your right to life and personal liberty, while Article 14 guarantees equality before the law. If you believe your fundamental rights have been violated, you can approach the High Court or Supreme Court directly through a writ petition. The courts have the power to issue various writs including habeas corpus, mandamus, and certiorari to protect your rights. The process is designed to be accessible and effective in addressing violations of fundamental rights.",
                "key_elements": ["fundamental rights", "constitutional remedies", "writ jurisdiction", "judicial review", "legal procedures"]
            },
            "civil_law": {
                "example": "I understand you're dealing with a civil dispute. Under Indian civil law, you have several legal remedies available depending on the nature of your case. For breach of contract, you can seek specific performance or damages. The Civil Procedure Code provides the framework for filing suits and the process of litigation. The court will examine the evidence, hear both parties, and make a decision based on the merits of the case. You have the right to legal representation, and the court can also order interim relief if needed. The process may take time, but it ensures a fair hearing of your case.",
                "key_elements": ["civil remedies", "contract law", "court procedures", "evidence", "legal representation"]
            }
        }
        
        # Update fallback summaries with more specific emergency guidance
        fallback_summaries = {
            "family_law": "If you are facing dowry harassment or domestic violence, your safety is the priority. You can immediately approach the nearest police station to file a complaint under Section 498A IPC. You can also contact the National Commission for Women helpline (7827170170) or your local women's protection cell. The police can help you get a protection order, and you can seek shelter at a women's protection home. Remember to document all incidents of harassment, including messages, photos of injuries, and witness statements. You have the right to live with dignity and safety, and the law provides strong protection against dowry harassment and domestic violence.",
            "consumer_rights": "If you are experiencing serious side effects from medicine, stop taking it immediately and seek medical attention. Contact your doctor or visit the nearest hospital emergency room. Keep the medicine package, prescription, and any remaining tablets as evidence. Report the adverse reaction to the nearest drug control authority and the manufacturer. You can also file a complaint with the consumer forum for compensation. The Drugs and Cosmetics Act provides protection against defective medicines, and you have the right to safe healthcare products. Document all symptoms, medical expenses, and communications with healthcare providers.",
            "criminal_law": "If you are arrested, you have the right to be informed of the grounds for arrest, to consult a lawyer, and to be produced before a magistrate within 24 hours. Remain calm, ask for your rights, and contact a trusted person or legal aid immediately.",
            "property_law": "If you are facing an urgent property dispute, gather all your documents and approach the local police or a civil court for immediate relief. Do not sign any documents under pressure and seek legal advice as soon as possible.",
            "employment_law": "If you are being harassed or unfairly treated at work, document all incidents and contact your HR department or a labor officer. You have the right to a safe workplace and can file a complaint with the labor commissioner or police if needed.",
            "constitutional_law": "If your fundamental rights are being violated, you can approach the High Court or Supreme Court for immediate protection through a writ petition. Legal aid is available if you cannot afford a lawyer.",
            "civil_law": "If you are facing an urgent civil dispute, such as breach of contract or property seizure, you can approach the civil court for interim relief or an injunction. Gather all evidence and seek legal advice promptly."
        }

        # Get relevant context with domain-specific keywords
        try:
            logger.info(f"Retrieving context for {domain} domain (Emergency: {is_emergency}, Personal: {is_personal_query})")
            context = rag_manager.get_relevant_context(
                query + " " + domain_context_keywords.get(domain, ""),
                k=3  # Get more context documents
            )
            # If context is empty or only contains legal code, use fallback summary
            def is_only_legal_code(text):
                return text.strip().startswith("Section") or text.strip().startswith("---") or len(text.strip()) < 50
            if not context or is_only_legal_code(context):
                logger.info(f"Using fallback summary for {domain}")
                context = fallback_summaries.get(domain, "Please consult a lawyer for immediate assistance.")
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            context = fallback_summaries.get(domain, "Please consult a lawyer for immediate assistance.")

        # Create prompt with enhanced emergency and personal handling
        prompt = f"""You are a helpful legal assistant providing clear, practical advice about Indian law. {'This is an emergency situation that requires immediate attention.' if is_emergency else ''} {'The user is asking about their personal situation.' if is_personal_query else ''} Provide a natural, empathetic response that directly addresses their specific situation.\n\nUser's Question: {query}\n\nLegal Information:\n{context}\n\nCRITICAL INSTRUCTIONS:\n1. {'Start with an immediate action step if this is an emergency situation.' if is_emergency else 'Start with an empathetic statement acknowledging their situation.'}\n2. {'For emergencies, prioritize immediate safety and legal rights.' if is_emergency else 'Explain their legal options and steps in a natural, flowing way.'}\n3. Include specific laws and sections that apply to their case\n4. {'For emergencies, provide clear next steps and immediate actions they can take.' if is_emergency else 'Explain the process in practical terms they can understand.'}\n5. Write in a single paragraph using complete sentences\n6. Use natural transitions between different aspects\n7. NEVER use lists, numbers, or any kind of markers\n8. NEVER say you cannot provide legal advice\n9. {'For emergencies, include relevant emergency contact numbers or immediate resources.' if is_emergency else ''}\n10. NEVER repeat these instructions in your response\n\nExample of good format for {domain.replace('_', ' ')}:\n{domain_templates[domain]['example']}\n\nWrite a helpful, practical response that directly addresses the user's specific situation and explains their legal options in clear, natural language. {'Focus on immediate actions and safety if this is an emergency.' if is_emergency else ''} Make sure to mention specific laws and procedures that apply to their case."""

        # Generate response with error handling
        try:
            logger.info(f"Starting response generation for {domain} domain")
            logger.info(f"Using model: {model_pipelines[language]['primary']['model']}")
            
            # First attempt with parameters optimized for natural, user-focused responses
            response = await generate_response(
                prompt=prompt,
                language=language,
                temperature=0.6,  # Balanced temperature for natural but focused language
                top_p=0.85,      # Balanced top_p for natural language
                max_new_tokens=400,  # Reduced to avoid sequence length issues
                do_sample=True
            )
            
            # Check if response is too short or contains disclaimers
            if len(response.split()) < 50 or "unable to provide" in response.lower() or "cannot provide" in response.lower():
                logger.info("Regenerating response to be more helpful and specific")
                response = await generate_response(
                    prompt=prompt + "\nCRITICAL: Write a natural, helpful response that explains the specific laws and procedures that apply to their case. Start with an empathetic statement, then explain their legal options and the steps they can take. Include specific sections of relevant laws.",
                    language=language,
                    temperature=0.5,  # Lower temperature for more focused response
                    top_p=0.8,       # More focused language
                    max_new_tokens=400,  # Keep within model limits
                    do_sample=True
                )
        
        except ValueError as ve:
            logger.error(f"Value error during response generation: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to generate response: {str(e)}")
        
        # Enhanced validation checks with stronger anti-list enforcement
        validation_issues = {
            "list_markers": [
                "- ", "• ", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
                ":", "---", "•", "*", "→", ">", "–", "—", "•", "○", "●", "■", "□",
                "first", "second", "third", "fourth", "fifth", "sixth", "finally",
                "types of", "categories", "kinds of", "forms of", "ways to",
                "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
                "constitutional rights:", "arrest rights:", "trial rights:",
                "right to", "rights include", "rights are", "rights of"
            ],
            "instruction_phrases": [
                "write in", "natural language", "complete sentences",
                "focus on", "legal concepts", "simple terms",
                "do not include", "disclaimers", "meta-commentary",
                "do not repeat", "instructions", "provided information",
                "doesn't fully answer", "explain what information",
                "critical instructions", "write a clear", "example of"
            ],
            "off_topic_indicators": [
                "in addition", "furthermore", "moreover",
                "it is also important to note", "other legal aspects",
                "related to this", "in the context of",
                "what are the laws regarding", "types of",
                "who can be liable", "grounds for",
                "general information", "broadly speaking",
                "in general", "typically", "usually"
            ]
        }
        
        # Length validation checks
        length_checks = {
            "too_short": len(response.split()) < 30,  # Increased minimum length
            "too_long": len(response.split()) > 200,  # Increased maximum length
            "multiple_paragraphs": len(response.split("\n")) > 1,
            "too_many_chars": len(response) > 1000  # Increased character limit
        }
        
        # Check for any issues
        has_issues = False
        
        # Check string-based validations
        for issue_type, checks in validation_issues.items():
            if any(phrase.lower() in response.lower() for phrase in checks):
                has_issues = True
                logger.warning(f"Response contains {issue_type} - regenerating")
                break
        
        # Check length-based validations
        if not has_issues:
            for check_name, check_result in length_checks.items():
                if check_result:
                    has_issues = True
                    logger.warning(f"Response has {check_name} - regenerating")
                    break
        
        if has_issues:
            try:
                # Try one more time with stricter parameters
                logger.info("Regenerating response with stricter parameters")
                response = await generate_response(
                    prompt=prompt + "\nCRITICAL: Write a single paragraph using ONLY complete sentences. NEVER use lists, numbers, or any kind of markers. Focus ONLY on answering the specific question asked. Start with a clear statement about the main legal provisions and then explain all rights and procedures in a flowing narrative.",
                    language=language,
                    temperature=0.2,  # Lower temperature for more focused response
                    top_p=0.7,       # More focused language
                    max_new_tokens=400,  # Increased for complete answer
                    do_sample=True
                )
                
                # If still has issues but response is too short, try one more time
                if len(response.split()) < 30:
                    logger.info("Response too short, trying one more time with balanced parameters")
                    response = await generate_response(
                        prompt=prompt + "\nCRITICAL: Write a comprehensive single paragraph that explains all legal rights and procedures in a natural, flowing narrative. Use ONLY complete sentences with no lists or markers. Start with the main legal provisions and then explain all aspects in detail.",
                        language=language,
                        temperature=0.3,  # Balanced temperature
                        top_p=0.8,       # More natural language
                        max_new_tokens=500,  # Allow enough for complete answer
                        do_sample=True
                    )
            except Exception as e:
                logger.error(f"Error during response regeneration: {str(e)}", exc_info=True)
                # If regeneration fails, return the original response if it's valid
                if len(response.split()) >= 30 and not any(phrase.lower() in response.lower() for phrase in validation_issues["list_markers"]):
                    logger.info("Using original response after regeneration failure")
                else:
                    raise
        
        return {
            "response": response,
            "language": language
        }
        
    except Exception as e:
        logger.error(f"Error in ask_legal_response: {str(e)}", exc_info=True)
        # Return a more specific error message based on the type of error
        if isinstance(e, ValueError):
            return {
                "response": str(e),
                "language": language
            }
        elif "context" in str(e).lower():
            return {
                "response": "I apologize, but I couldn't find relevant legal information to answer your question. Please try rephrasing your question or asking about a different legal topic.",
                "language": language
            }
        else:
            return {
                "response": "I apologize, but I encountered an unexpected error while processing your request. Please try again in a moment.",
                "language": language
            }

async def generate_fir_pdf(name: str, location: str, details: str) -> str:
    try:
        # Generate unique filename with timestamp
        filename = f"FIR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        logger.info(f"Generating FIR PDF with filename: {filename}")
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Set font styles
        pdf.set_font("Arial", 'B', 16)  # Bold for headers
        pdf.set_font("Arial", '', 12)   # Regular for content
        
        # Header
        pdf.cell(0, 10, "FIRST INFORMATION REPORT (FIR)", ln=True, align='C')
        pdf.cell(0, 10, "Police Station: _________________", ln=True, align='C')
        pdf.cell(0, 10, f"District: _________________", ln=True, align='C')
        pdf.cell(0, 10, f"State: _________________", ln=True, align='C')
        pdf.ln(5)
        
        # FIR Number and Date
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "FIR No.:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "_________________", ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Date of Registration:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, datetime.now().strftime("%d-%m-%Y"), ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Time of Registration:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, datetime.now().strftime("%H:%M:%S"), ln=True)
        pdf.ln(5)
        
        # Complainant Details
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "1. COMPLAINANT DETAILS", ln=True)
        pdf.set_font("Arial", '', 12)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Name:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, name, ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Address:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, location)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Contact No.:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "_________________", ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Email:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "_________________", ln=True)
        pdf.ln(5)
        
        # Incident Details
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "2. INCIDENT DETAILS", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(60, 10, "Date of Incident:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "_________________", ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Time of Incident:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "_________________", ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(60, 10, "Place of Incident:", ln=0)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, location)
        
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "Detailed Description:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, details)
        pdf.ln(5)
        
        # Section of Law
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "3. SECTION OF LAW", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "To be filled by the Police", ln=True)
        pdf.ln(5)
        
        # Witness Details
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "4. WITNESS DETAILS", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "Name: _________________", ln=True)
        pdf.cell(0, 10, "Address: _________________", ln=True)
        pdf.cell(0, 10, "Contact: _________________", ln=True)
        pdf.ln(5)
        
        # Property Details (if applicable)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "5. PROPERTY DETAILS (IF ANY)", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, "Description: _________________", ln=True)
        pdf.cell(0, 10, "Estimated Value: _________________", ln=True)
        pdf.ln(5)
        
        # Signature Section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "6. SIGNATURES", ln=True)
        pdf.set_font("Arial", '', 12)
        
        pdf.cell(90, 10, "Complainant's Signature", ln=0, align='C')
        pdf.cell(90, 10, "Police Officer's Signature", ln=True, align='C')
        pdf.cell(90, 10, "_________________", ln=0, align='C')
        pdf.cell(90, 10, "_________________", ln=True, align='C')
        
        pdf.cell(90, 10, "Name: " + name, ln=0, align='C')
        pdf.cell(90, 10, "Name: _________________", ln=True, align='C')
        
        pdf.cell(90, 10, "Date: " + datetime.now().strftime("%d-%m-%Y"), ln=0, align='C')
        pdf.cell(90, 10, "Designation: _________________", ln=True, align='C')
        
        # Save PDF to static directory
        try:
            # Ensure static directory exists
            static_dir = Path("static")
            static_dir.mkdir(exist_ok=True)
            logger.info(f"Static directory path: {static_dir.absolute()}")
            
            # Generate PDF bytes
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            logger.info(f"Generated PDF bytes, size: {len(pdf_bytes)}")
            
            # Save PDF file
            pdf_path = static_dir / filename
            pdf_path.write_bytes(pdf_bytes)
            logger.info(f"PDF saved successfully at: {pdf_path.absolute()}")
            
            # Verify file exists
            if pdf_path.exists():
                logger.info(f"Verified PDF file exists at: {pdf_path.absolute()}")
                return filename
            else:
                raise Exception(f"PDF file was not created at: {pdf_path.absolute()}")
                
        except Exception as e:
            logger.error(f"Error saving PDF: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error saving PDF: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating PDF: {str(e)}"
        )

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static directory for PDFs
app.mount("/static", StaticFiles(directory="static"), name="static")

# API endpoints
@app.post("/ask")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def ask_legal_bot(
    request: Request,
    data: LegalQuery,
    api_key: str = Depends(verify_api_key)
):
    """Enhanced legal bot endpoint with conversation support."""
    # Check if application is ready
    if not app.state.ready:
        logger.warning("Request received while application is not ready")
        raise HTTPException(
            status_code=503,
            detail="Service is initializing, please try again in a moment"
        )
        
    try:
        logger.info(f"Received query: {data.query} in {data.language}")
        
        # Validate model pipelines are loaded
        if not model_pipelines or data.language not in model_pipelines:
            logger.error("Model pipelines not properly initialized")
            raise HTTPException(
                status_code=503,
                detail="Service is initializing, please try again in a moment"
            )
            
        # Check if primary model is available
        if "primary" not in model_pipelines[data.language]:
            logger.error(f"Primary model not available for {data.language}")
            raise HTTPException(
                status_code=503,
                detail="Language model is not available, please try again later"
            )
            
        # Log pipeline status
        logger.info(f"Using pipeline: {model_pipelines[data.language]['primary']['model']}")
        
        response = await ask_legal_response(
            query=data.query,
            language=data.language
        )
            
        if not response or "response" not in response:
            logger.error("Invalid response format from ask_legal_response")
            raise HTTPException(
                status_code=500,
                detail="Invalid response format"
            )
            
        logger.info("Successfully generated response")
        return {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as ve:
        logger.error(f"Value error in ask_legal_response: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error in ask_legal_response: {str(e)}", exc_info=True)
        # Check for specific error types
        if "CUDA" in str(e):
            raise HTTPException(
                status_code=503,
                detail="GPU memory error - please try again in a moment"
            )
        elif "timeout" in str(e).lower():
            raise HTTPException(
                status_code=504,
                detail="Request timed out - please try again"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

@app.post("/generate-pdf")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def generate_pdf(
    request: Request,
    data: DocumentData,
    api_key: str = Depends(verify_api_key)
):
    try:
        logger.info(f"Received PDF generation request for: {data.name}")
        filename = await generate_fir_pdf(data.name, data.location, data.details)
        logger.info(f"PDF generated successfully: {filename}")
        return {"file": filename}
    except Exception as e:
        logger.error(f"Error in /generate-pdf endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating PDF: {str(e)}"
        )

@app.post("/predict")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def predict(
    request: Request,
    data: LegalQuery,
    api_key: str = Depends(verify_api_key)
):
    """
    Test endpoint for getting legal assistant responses without WhatsApp integration.
    Useful for testing and development.
    """
    try:
        response = await ask_legal_response(data.query, data.language)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def split_message(text: str, max_length: int = 1500) -> List[str]:
    """Split a long message into chunks that fit within Twilio's character limit."""
    if len(text) <= max_length:
        return [text]
    
    # Split by sentences to keep messages readable
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add period back to the sentence
        sentence = sentence + '. '
        
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def handle_fir_command(message: WhatsAppMessage) -> Tuple[str, Optional[str]]:
    """Handle FIR generation command from WhatsApp. Returns (response_text, pdf_url)."""
    try:
        # Extract FIR details from the message (preserve original casing)
        text = message.body
        # Split by comma but preserve commas within quotes
        parts = []
        current_part = ""
        in_quotes = False
        for char in text[4:].strip():
            if char == '"':
                in_quotes = not in_quotes
                current_part += char
            elif char == ',' and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        if current_part:
            parts.append(current_part.strip())
        if len(parts) < 3:
            response_text = truncate_response("Please provide all required details: name: [your name], location: [incident location], details: [incident details]")
            await send_whatsapp_message(message.sender, response_text)
            return response_text, None
        # Extract information with improved parsing (preserve casing)
        name = ""
        location = ""
        details = ""
        found_details = False
        for idx, part in enumerate(parts):
            part_strip = part.strip()
            if not found_details:
                if 'name:' in part_strip.lower():
                    name = part_strip.split(':', 1)[1].strip()
                elif 'location:' in part_strip.lower():
                    location = part_strip.split(':', 1)[1].strip()
                elif 'details:' in part_strip.lower():
                    # Join all remaining parts for details
                    details = part_strip.split(':', 1)[1].strip()
                    if idx + 1 < len(parts):
                        details += ', ' + ', '.join(p.strip() for p in parts[idx+1:])
                    found_details = True
            # If already found details, skip further parsing
        logger.info(f"Extracted FIR details - Name: {name}, Location: {location}, Details length: {len(details)}")
        if not name or not location or not details:
            response_text = truncate_response("Missing required information. Please provide name: [your name], location: [incident location], details: [incident details]")
            await send_whatsapp_message(message.sender, response_text)
            return response_text, None
        # Generate FIR PDF
        try:
            filename = f"FIR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            logger.info(f"Starting FIR PDF generation with filename: {filename}")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / filename
                logger.info(f"Creating temporary PDF at: {temp_path}")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, "FIRST INFORMATION REPORT (FIR)", ln=True, align='C')
                pdf.cell(0, 10, "Police Station: _________________", ln=True, align='C')
                pdf.cell(0, 10, f"District: _________________", ln=True, align='C')
                pdf.cell(0, 10, f"State: _________________", ln=True, align='C')
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(60, 10, "FIR No.:", ln=0)
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, "_________________", ln=True)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(60, 10, "Date of Registration:", ln=0)
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, datetime.now().strftime("%d-%m-%Y"), ln=True)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(60, 10, "Time of Registration:", ln=0)
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, datetime.now().strftime("%H:%M:%S"), ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "1. COMPLAINANT DETAILS", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(60, 10, "Name:", ln=0)
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, name, ln=True)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(60, 10, "Address:", ln=0)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, location)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "2. INCIDENT DETAILS", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, details)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "3. SIGNATURES", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.cell(90, 10, "Complainant's Signature", ln=0, align='C')
                pdf.cell(90, 10, "Police Officer's Signature", ln=True, align='C')
                pdf.cell(90, 10, "_________________", ln=0, align='C')
                pdf.cell(90, 10, "_________________", ln=True, align='C')
                pdf.cell(90, 10, f"Name: {name}", ln=0, align='C')
                pdf.cell(90, 10, "Name: _________________", ln=True, align='C')
                pdf.cell(90, 10, f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=0, align='C')
                pdf.cell(90, 10, "Designation: _________________", ln=True, align='C')
                try:
                    pdf.output(str(temp_path))
                    logger.info(f"PDF generated successfully at temporary location: {temp_path}")
                    if not temp_path.exists():
                        raise Exception("Temporary PDF file was not created")
                    if temp_path.stat().st_size == 0:
                        raise Exception("Temporary PDF file is empty")
                    static_dir = Path("static")
                    static_dir.mkdir(exist_ok=True)
                    logger.info(f"Static directory path: {static_dir.absolute()}")
                    final_path = static_dir / filename
                    shutil.copy2(temp_path, final_path)
                    logger.info(f"PDF copied to final location: {final_path}")
                    if not final_path.exists():
                        raise Exception(f"PDF file was not copied to final location: {final_path}")
                    if final_path.stat().st_size == 0:
                        raise Exception(f"Final PDF file is empty: {final_path}")
                    pdf_url = f"{settings.PUBLIC_BASE_URL}/static/{filename}"
                    logger.info(f"Generated PDF URL: {pdf_url}")
                    response_text = truncate_response("✓ FIR document generated successfully! Please find your FIR document attached. After reviewing, please visit your nearest police station with this document.")
                    await send_whatsapp_message(message.sender, response_text)
                    await asyncio.sleep(2)
                    try:
                        media_message = twilio_client.messages.create(
                            body="Your FIR document",
                            from_=f'whatsapp:{settings.TWILIO_NUMBER}',
                            to=f'whatsapp:{message.sender}',
                            media_url=[pdf_url]
                        )
                        logger.info(f"PDF message sent successfully - SID: {media_message.sid}")
                        return response_text, pdf_url
                    except Exception as e:
                        logger.error(f"Error sending PDF message: {str(e)}", exc_info=True)
                        # If media message fails, send the link as a text message
                        link_message = f"Your FIR document is ready. Download it here: {pdf_url}"
                        await send_whatsapp_message(message.sender, link_message)
                        return response_text, None
                except Exception as e:
                    logger.error(f"Error in PDF generation/saving process: {str(e)}", exc_info=True)
                    response_text = truncate_response("Sorry, there was an error generating your FIR document. Please try again later.")
                    await send_whatsapp_message(message.sender, response_text)
                    return response_text, None
        except Exception as e:
            logger.error(f"Error in PDF generation: {str(e)}", exc_info=True)
            response_text = truncate_response("Sorry, there was an error generating your FIR document. Please try again later.")
            await send_whatsapp_message(message.sender, response_text)
            return response_text, None
    except Exception as e:
        logger.error(f"Error processing FIR command: {str(e)}", exc_info=True)
        response_text = truncate_response("Sorry, I couldn't process your FIR request. Please use the format: /fir name: [your name], location: [incident location], details: [incident details]")
        await send_whatsapp_message(message.sender, response_text)
        return response_text, None

def truncate_response(text: str, max_length: int = 1000) -> str:
    """Truncate response text to max_length characters, preserving sentence boundaries."""
    if len(text) <= max_length:
        return text
        
    # Find the last complete sentence within max_length
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    if last_period > 0:
        return truncated[:last_period + 1]
    return truncated

@app.post("/webhook", response_class=PlainTextResponse)
@limiter.limit(f"10/minute")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    start_time = datetime.utcnow()
    try:
        # Get form data
        form_data = await request.form()
        logger.info(f"Received Twilio webhook payload (form_data): {dict(form_data)}")
        
        # Validate message
        try:
            message = await validate_whatsapp_message(form_data)
            logger.info(f"Validated WhatsApp message: {message.dict()}")
        except HTTPException as e:
            logger.error(f"Message validation failed: {str(e)}")
            return PlainTextResponse(str(e.detail), status_code=e.status_code)

        # Log request in background
        background_tasks.add_task(
            log_webhook_request,
            request,
            message,
            (datetime.utcnow() - start_time).total_seconds()
        )

        # Process the message
        if message.body.lower().startswith('/fir'):
            # Handle FIR generation command
            try:
                # Extract FIR details from the message (preserve original casing)
                text = message.body
                # Split by comma but preserve commas within quotes
                parts = []
                current_part = ""
                in_quotes = False
                for char in text[4:].strip():
                    if char == '"':
                        in_quotes = not in_quotes
                        current_part += char
                    elif char == ',' and not in_quotes:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                if current_part:
                    parts.append(current_part.strip())
                if len(parts) < 3:
                    response_text = truncate_response("Please provide all required details: name: [your name], location: [incident location], details: [incident details]")
                    await send_whatsapp_message(message.sender, response_text)
                    return PlainTextResponse("", status_code=200)
                # Extract information with improved parsing (preserve casing)
                name = ""
                location = ""
                details = ""
                found_details = False
                for idx, part in enumerate(parts):
                    part_strip = part.strip()
                    if not found_details:
                        if 'name:' in part_strip.lower():
                            name = part_strip.split(':', 1)[1].strip()
                        elif 'location:' in part_strip.lower():
                            location = part_strip.split(':', 1)[1].strip()
                        elif 'details:' in part_strip.lower():
                            # Join all remaining parts for details
                            details = part_strip.split(':', 1)[1].strip()
                            if idx + 1 < len(parts):
                                details += ', ' + ', '.join(p.strip() for p in parts[idx+1:])
                            found_details = True
                    # If already found details, skip further parsing
                logger.info(f"Extracted FIR details - Name: {name}, Location: {location}, Details length: {len(details)}")
                if not name or not location or not details:
                    response_text = truncate_response("Missing required information. Please provide name: [your name], location: [incident location], details: [incident details]")
                    await send_whatsapp_message(message.sender, response_text)
                    return PlainTextResponse("", status_code=200)
                # Generate FIR PDF
                try:
                    filename = f"FIR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    logger.info(f"Starting FIR PDF generation with filename: {filename}")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir) / filename
                        logger.info(f"Creating temporary PDF at: {temp_path}")
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", 'B', 16)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(0, 10, "FIRST INFORMATION REPORT (FIR)", ln=True, align='C')
                        pdf.cell(0, 10, "Police Station: _________________", ln=True, align='C')
                        pdf.cell(0, 10, f"District: _________________", ln=True, align='C')
                        pdf.cell(0, 10, f"State: _________________", ln=True, align='C')
                        pdf.ln(5)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(60, 10, "FIR No.:", ln=0)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(0, 10, "_________________", ln=True)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(60, 10, "Date of Registration:", ln=0)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(0, 10, datetime.now().strftime("%d-%m-%Y"), ln=True)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(60, 10, "Time of Registration:", ln=0)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(0, 10, datetime.now().strftime("%H:%M:%S"), ln=True)
                        pdf.ln(5)
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(0, 10, "1. COMPLAINANT DETAILS", ln=True)
                        pdf.set_font("Arial", '', 12)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(60, 10, "Name:", ln=0)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(0, 10, name, ln=True)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(60, 10, "Address:", ln=0)
                        pdf.set_font("Arial", '', 12)
                        pdf.multi_cell(0, 10, location)
                        pdf.ln(5)
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(0, 10, "2. INCIDENT DETAILS", ln=True)
                        pdf.set_font("Arial", '', 12)
                        pdf.multi_cell(0, 10, details)
                        pdf.ln(5)
                        pdf.set_font("Arial", 'B', 14)
                        pdf.cell(0, 10, "3. SIGNATURES", ln=True)
                        pdf.set_font("Arial", '', 12)
                        pdf.cell(90, 10, "Complainant's Signature", ln=0, align='C')
                        pdf.cell(90, 10, "Police Officer's Signature", ln=True, align='C')
                        pdf.cell(90, 10, "_________________", ln=0, align='C')
                        pdf.cell(90, 10, "_________________", ln=True, align='C')
                        pdf.cell(90, 10, f"Name: {name}", ln=0, align='C')
                        pdf.cell(90, 10, "Name: _________________", ln=True, align='C')
                        pdf.cell(90, 10, f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=0, align='C')
                        pdf.cell(90, 10, "Designation: _________________", ln=True, align='C')
                        try:
                            pdf.output(str(temp_path))
                            logger.info(f"PDF generated successfully at temporary location: {temp_path}")
                            if not temp_path.exists():
                                raise Exception("Temporary PDF file was not created")
                            if temp_path.stat().st_size == 0:
                                raise Exception("Temporary PDF file is empty")
                            static_dir = Path("static")
                            static_dir.mkdir(exist_ok=True)
                            logger.info(f"Static directory path: {static_dir.absolute()}")
                            final_path = static_dir / filename
                            shutil.copy2(temp_path, final_path)
                            logger.info(f"PDF copied to final location: {final_path}")
                            if not final_path.exists():
                                raise Exception(f"PDF file was not copied to final location: {final_path}")
                            if final_path.stat().st_size == 0:
                                raise Exception(f"Final PDF file is empty: {final_path}")
                            pdf_url = f"{settings.PUBLIC_BASE_URL}/static/{filename}"
                            logger.info(f"Generated PDF URL: {pdf_url}")
                            response_text = truncate_response("✓ FIR document generated successfully! Please find your FIR document attached. After reviewing, please visit your nearest police station with this document.")
                            await send_whatsapp_message(message.sender, response_text)
                            await asyncio.sleep(2)
                            try:
                                media_message = twilio_client.messages.create(
                                    body="Your FIR document",
                                    from_=f'whatsapp:{settings.TWILIO_NUMBER}',
                                    to=f'whatsapp:{message.sender}',
                                    media_url=[pdf_url]
                                )
                                logger.info(f"PDF message sent successfully - SID: {media_message.sid}")
                                return PlainTextResponse("", status_code=200)
                            except Exception as e:
                                logger.error(f"Error sending PDF message: {str(e)}", exc_info=True)
                                # If media message fails, send the link as a text message
                                link_message = f"Your FIR document is ready. Download it here: {pdf_url}"
                                await send_whatsapp_message(message.sender, link_message)
                                return PlainTextResponse("", status_code=200)
                        except Exception as e:
                            logger.error(f"Error in PDF generation/saving process: {str(e)}", exc_info=True)
                            response_text = truncate_response("Sorry, there was an error generating your FIR document. Please try again later.")
                            await send_whatsapp_message(message.sender, response_text)
                            return PlainTextResponse("", status_code=200)
                except Exception as e:
                    logger.error(f"Error in PDF generation: {str(e)}", exc_info=True)
                    response_text = truncate_response("Sorry, there was an error generating your FIR document. Please try again later.")
                    await send_whatsapp_message(message.sender, response_text)
                    return PlainTextResponse("", status_code=200)
            except Exception as e:
                logger.error(f"Error processing FIR command: {str(e)}", exc_info=True)
                response_text = truncate_response("Sorry, I couldn't process your FIR request. Please use the format: /fir name: [your name], location: [incident location], details: [incident details]")
                await send_whatsapp_message(message.sender, response_text)
                return PlainTextResponse("", status_code=200)
        else:
            # Handle regular messages with Groq API
            try:
                # Get response from legal assistant
                response = await ask_legal_response(
                    query=message.body,
                    language="english"  # Default to English for now
                )
                
                if response and "response" in response:
                    # Truncate and send the response back to the user
                    truncated_response = truncate_response(response["response"])
                    await send_whatsapp_message(message.sender, truncated_response)
                else:
                    # Send error message if no response
                    await send_whatsapp_message(
                        message.sender,
                        truncate_response("I apologize, but I couldn't process your request. Please try again.")
                    )
                    
            except Exception as e:
                logger.error(f"Error processing regular message: {str(e)}", exc_info=True)
                await send_whatsapp_message(
                    message.sender,
                    truncate_response("I apologize, but I encountered an error while processing your message. Please try again.")
                )
            
            return PlainTextResponse("", status_code=200)

    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}", exc_info=True)
        return PlainTextResponse("", status_code=200)

@app.post("/speak")
async def speak(text: str, language: str = "english"):
    # Map language to gTTS language code
    lang_code = "en" if language == "english" else "hi"
    tts = gTTS(text, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return StreamingResponse(audio_fp, media_type="audio/mpeg")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "english"):
    recognizer = sr.Recognizer()
    audio_bytes = await file.read()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)
    try:
        lang_code = "en-IN" if language == "english" else ("hi-IN" if language == "hindi" else "en-IN")
        text = recognizer.recognize_google(audio, language=lang_code)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask_with_voice")
async def ask_with_voice(request: LegalQuery):
    # Get the text answer
    answer = await ask_legal_response(request.query, request.language)
    # Generate audio
    lang_code = "en" if request.language == "english" else ("hi" if request.language == "hindi" else "hi")
    tts = gTTS(answer, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    audio_base64 = base64.b64encode(audio_fp.read()).decode("utf-8")
    return {"text": answer, "audio_base64": audio_base64}

# Cleanup task
@app.on_event("startup")
async def schedule_cleanup():
    async def cleanup_loop():
        while True:
            try:
                file_manager.cleanup_old_files()
                await asyncio.sleep(24 * 60 * 60)  # Run daily
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    asyncio.create_task(cleanup_loop())

async def validate_whatsapp_message(form_data: Dict[str, Any]) -> WhatsAppMessage:
    """Validate and sanitize WhatsApp message data."""
    try:
        msg_body = form_data.get("Body", "").strip()
        sender = form_data.get("From", "").strip()
        num_media = int(form_data.get("NumMedia", "0"))
        
        if not msg_body and num_media == 0:
            raise ValueError("Message body is required if no media is present")
        if not sender:
            raise ValueError("Sender information is missing")
            
        # Basic sanitization - preserve forward slashes and colons for commands
        msg_body = re.sub(r'[^\w\s.,!?:/-]', '', msg_body)  # Keep forward slashes and colons for commands
        sender = re.sub(r'[^\d+]', '', sender)  # Keep only digits and + for phone numbers
        
        message_type = "media" if num_media > 0 else "text"
        media_url = form_data.get("MediaUrl0") if num_media > 0 else None
        
        return WhatsAppMessage(
            body=msg_body,
            sender=sender,
            message_type=message_type,
            media_url=media_url
        )
    except Exception as e:
        logger.error(f"Message validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

async def log_webhook_request(request: Request, message: WhatsAppMessage, response_time: float):
    """Log webhook request details for monitoring and debugging."""
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "method": request.method,
        "url": str(request.url),
        "client_host": request.client.host if request.client else None,
        "message_type": message.message_type,
        "sender": message.sender,
        "body_length": len(message.body),
        "response_time": response_time,
        "user_agent": request.headers.get("user-agent"),
        "twilio_signature": request.headers.get("x-twilio-signature")
    }
    logger.info(f"Webhook request: {log_data}")

class UpdateResponse(BaseModel):
    status: str
    message: str
    new_documents_added: int

@app.post("/update-vector-store", response_model=UpdateResponse)
@limiter.limit("5/minute")  # Limit to 5 updates per minute
async def update_vector_store(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Update the vector store with any new files in the legal_docs directory."""
    try:
        # Call the update method
        rag_manager.update_vector_store_with_new_files()
        
        # Get the number of documents in the vector store
        if rag_manager.vector_store and hasattr(rag_manager.vector_store, "docstore"):
            doc_count = len(rag_manager.vector_store.docstore._dict)
        else:
            doc_count = 0
            
        return UpdateResponse(
            status="success",
            message="Vector store updated successfully",
            new_documents_added=doc_count
        )
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating vector store: {str(e)}"
        ) 

if __name__ == "__main__":
    # Configure uvicorn to use our reload protection
    config = uvicorn.Config(
        "specter_legal_assistant:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_delay=1,
        reload_includes=["*.py"],
        reload_excludes=["*.pyc", "*.pyo", "*.pyd"],
        log_level="info"
    )
    
    # Add reload callback
    def on_reload():
        if not should_reload():
            logger.warning("Reload blocked by protection")
            return False
        return True
    
    config.reload_callback = on_reload
    server = uvicorn.Server(config)
    server.run() 