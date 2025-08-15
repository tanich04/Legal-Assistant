# SPECTER Legal Assistant

SPECTER is an AI-powered legal assistant that automates the generation of First Information Reports (FIRs) and delivers them to users via WhatsApp using Twilio. It also provides legal query handling using advanced language models.

## Features
- WhatsApp-based legal assistance
- Automated FIR PDF generation
- Secure document delivery via public URLs
- AI/NLP-powered legal query responses
- Modular FastAPI backend

## Project Structure
- `specter_legal_assistant/` - Main backend code
- `requirements.txt` - Python dependencies
- `static/` - Directory for generated PDFs

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ak47shat/SPECTER.git
   cd SPECTER
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment:**
   - Create a `.env` file with your Twilio and server settings.
4. **Run the server:**
   ```bash
   uvicorn specter_legal_assistant:app --reload
   ```
5. **Expose your server (for WhatsApp integration):**
   - Use [ngrok](https://ngrok.com/) or similar to create a public URL.

## Usage
- Send `/fir name: <Name>, location: <Location>, details: <Details>` via WhatsApp to the configured Twilio number.
- For other legal queries, send a message directly.

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
