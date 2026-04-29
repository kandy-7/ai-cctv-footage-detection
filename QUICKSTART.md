# Quick Start Guide

## 🎯 Your Project Has Been Updated!

I've completely rebuilt your AI CCTV footage detection system with a beautiful, modern frontend to replace Streamlit. Here's what's new:

## ✨ What's Been Built

### Frontend (NEW!)
- **Modern UI**: Black & orange theme with glassmorphism effects
- **Drag & Drop Upload**: Intuitive video upload interface  
- **Natural Language Queries**: Ask questions in plain English
- **4K Video Player**: View enhanced video clips
- **Responsive Design**: Works on desktop and mobile

### Backend (Existing - Enhanced)
- FastAPI REST API
- YOLOv8 object detection
- Video processing pipeline
- 4K enhancement engine
- SQLite event database

## 🚀 How to Run

### Option 1: Automated Setup (Recommended)

1. **First Time Setup**:
   ```bash
   # Double-click setup.bat or run:
   setup.bat
   ```
   This will:
   - Create Python virtual environment
   - Install all Python dependencies
   - Install Node.js dependencies

2. **Start the Application**:
   ```bash
   # Double-click start.bat or run:
   start.bat
   ```
   This will start both servers automatically!

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Access the Application

Once running:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## 📝 How to Use

1. **Upload a Video**
   - Drag & drop your CCTV footage into the upload area
   - Or click to browse and select a video file
   - Wait for processing to complete (background task)

2. **Query the Footage**
   - Type your question in plain English
   - Example: "Show me when a person appears"
   - Click the Search button or press Enter

3. **View Results**
   - If found, a 4K enhanced clip will play automatically
   - Shows timestamp and number of occurrences
   - Download the clip using the download button

## 🎨 UI Features

- **Glassmorphism Design**: Modern frosted glass effect
- **Smooth Animations**: Professional transitions
- **Status Indicators**: Live processing feedback
- **Example Queries**: Quick-start chips for common searches
- **Responsive Layout**: Adapts to any screen size

## 🔧 Technical Details

**Frontend Stack:**
- Vite (Build tool)
- TypeScript (Type safety)
- Modern CSS (Glassmorphism, animations)

**Backend Stack:**
- FastAPI (REST API)
- YOLOv8 (Object detection)
- OpenCV (Video processing)
- SQLAlchemy (Database)

## 📦 File Structure

```
Ai cctv footage detection/
├── backend/
│   ├── main.py          # API endpoints
│   ├── processing.py    # Video processing
│   ├── search.py        # Query parsing
│   ├── enhancement.py   # 4K upscaling
│   ├── database.py      # Data models
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.ts     # Application logic
│   │   └── style.css   # Styling
│   ├── index.html
│   └── package.json
├── setup.bat           # Automated setup
├── start.bat           # Start servers
└── README.md           # Full documentation
```

## 🐛 Troubleshooting

### Backend Not Starting?
- Make sure Python 3.8+ is installed
- Activate virtual environment: `venv\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt`

### Frontend Not Starting?
- Make sure Node.js 16+ is installed
- Install dependencies: `npm install` (in frontend folder)

### CORS Errors?
- Backend must be running on port 8001
- Frontend must be running on port 5173

### Video Not Playing?
- Check browser console for errors
- Ensure backend processed the video successfully
- Verify video format is supported (MP4 recommended)

## 🎯 Next Steps

1. Run `setup.bat` to install all dependencies
2. Run `start.bat` to launch the application
3. Open http://localhost:5173 in your browser
4. Upload a test video and try some queries!

## 💡 Tips

- Use Chrome or Edge for best compatibility
- First upload will download YOLOv8 model (~6MB)
- Processing time depends on video length
- Queries are case-insensitive
- Try using synonyms (e.g., "human" = "person")

---

**Need help?** Check the full README.md for detailed documentation!
