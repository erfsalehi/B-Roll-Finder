# B-Roll Finder

B-Roll Finder is an AI-driven tool for finding and downloading relevant B-roll footage for your video projects.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) installed and added to your system PATH.

### 2. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. API Configuration
This project requires API keys from Groq, Pexels, and Pixabay. 

1. Create a `.env` file in the root directory (or copy `.env.example`).
2. Add your keys to the `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   PEXELS_API_KEY=your_pexels_api_key_here
   PIXABAY_API_KEY=your_pixabay_api_key_here
   ```

#### Where to find API Keys:
- **Groq API Key**: 
  - Go to the [Groq Cloud Console](https://console.groq.com/).
  - Navigate to **API Keys** and click **Create API Key**.
- **Pexels API Key**:
  - Visit the [Pexels API page](https://www.pexels.com/api/).
  - Sign up/log in and click **Get Started** to generate your key.
- **Pixabay API Key**:
  - Visit the [Pixabay API Documentation](https://pixabay.com/api/docs/).
  - Once logged in, your API key will be visible in the "Parameters" section of the documentation.

### 4. Running the App
```bash
python app.py
```

## Security Note
**DO NOT** push your `.env` file or any `*Api.txt` files to Git. These are already included in `.gitignore` to protect your credentials.
