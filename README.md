# AI CCTV Footage Detection System

An intelligent CCTV footage analysis system that uses AI-powered object detection (YOLOv8) to analyze video content and answer natural language queries. The system can detect objects, people, and events in video footage and enhance video clips to 4K resolution.

## 🚀 Features

- **Video Upload & Processing**: Upload CCTV footage for AI analysis
- **Natural Language Queries**: Ask questions in plain English (e.g., "Show me when a person appears")
- **Object Detection**: Powered by YOLOv8 for accurate real-time detection
- **4K Video Enhancement**: AI-upscaled video clips for detected events
- **Event Database**: SQLite database storing all detected events with timestamps
- **Modern UI**: Beautiful black & orange themed interface with glassmorphism effects

## 📋 System Architecture

### Backend (FastAPI + Python)
- **FastAPI**: High-performance REST API
- **YOLOv8**: State-of-the-art object detection
- **OpenCV**: Video processing and enhancement
- **SQLAlchemy**: Database ORM for event storage
- **Super Resolution**: AI-based video upscaling

### Frontend (Vite + TypeScript)
- **Vite**: Lightning-fast development and build
- **TypeScript**: Type-safe frontend code
- **Modern CSS**: Glassmorphism and smooth animations
- **Responsive Design**: Works on desktop and mobile

## 🛠️ Installation

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## 🚀 Running the Application

### Start the Backend

From the `backend` directory:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

The API will be available at: `http://localhost:8001`

### Start the Frontend

From the `frontend` directory:
```bash
npm run dev
```

The application will be available at: `http://localhost:5173`

## 📖 Usage

1. **Upload Video**: 
   - Drag and drop your CCTV footage or click to browse
   - Supported formats: MP4, AVI, MOV
   - The system will automatically process the video in the background

2. **Query the Footage**:
   - Use natural language to describe what you're looking for
   - Examples:
     - "Show me when a person appears"
     - "Find cars in the video"
     - "Person with backpack"
     - "Show me cats"

3. **View Results**:
   - If the event is found, a 4K enhanced clip will be displayed
   - Shows the timestamp and number of occurrences
   - Download the clip for further analysis

## 🔧 API Endpoints

### Upload Video
```http
POST /upload
Content-Type: multipart/form-data

Returns: { "video_id": "uuid", "message": "Video uploaded and processing started." }
```

### Query Video
```http
POST /query
Content-Type: application/json

Body: {
  "video_id": "uuid",
  "query": "person walking",
  "threshold": 0.5
}

Returns: { 
  "found": true, 
  "video_url": "url_to_4k_clip",
  "events": 5,
  "timestamp": 12.5
}
```

### Health Check
```http
GET /

Returns: { "message": "CCTV AI System Ready" }
```

## 🎯 Supported Object Classes

The system uses YOLOv8's COCO dataset, which includes 80+ object classes:

- **People**: person, man, woman, human
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, bird, horse
- **Objects**: backpack, handbag, umbrella, bottle, chair, etc.

## 🔥 Advanced Features

### Video Enhancement
- Automatic 4K upscaling using Super Resolution
- Fallback to bicubic interpolation if SR model unavailable
- Clip extraction with configurable time window

### Smart Search
- Synonym mapping for natural queries
- Confidence threshold filtering
- Timestamp-based event locating
- Multiple event detection

## 📁 Project Structure

```
ai-cctv-footage-detection/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── processing.py        # Video processing & YOLOv8
│   ├── search.py            # Natural language query parsing
│   ├── enhancement.py       # 4K upscaling
│   ├── database.py          # SQLAlchemy models
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── main.ts         # TypeScript application
│   │   └── style.css       # Styling
│   ├── index.html          # HTML entry point
│   └── package.json        # Node dependencies
└── README.md
```

## 🎨 UI/UX Features

- **Glassmorphism Design**: Modern frosted glass effect
- **Black & Orange Theme**: Professional and eye-catching
- **Drag & Drop Upload**: Intuitive file upload
- **Live Status**: Real-time processing indicators
- **Smooth Animations**: Polished user experience
- **Responsive Layout**: Mobile and desktop optimized

## ⚙️ Configuration

### Backend Configuration
- Default port: 8001
- CORS: Enabled for all origins (adjust in production)
- Upload directory: `uploads/`
- Processed clips: `processed/`

### Frontend Configuration
- Default port: 5173
- API endpoint: `http://localhost:8001`

## 🐛 Troubleshooting

### Backend Issues
- **Model Download**: YOLOv8 model downloads automatically on first run
- **CUDA/GPU**: System works on CPU, GPU acceleration optional
- **Super Resolution**: If model not found, uses bicubic fallback

### Frontend Issues
- **CORS Errors**: Ensure backend is running on port 8001
- **Video Not Playing**: Check browser video codec support
- **Upload Fails**: Verify backend is running and accessible

## 🚧 Future Enhancements

- [ ] Real-time video streaming
- [ ] Multi-camera support
- [ ] Advanced analytics dashboard
- [ ] Export reports (PDF/CSV)
- [ ] User authentication
- [ ] Cloud storage integration
- [ ] Mobile app

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📧 Support

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using FastAPI, YOLOv8, and Vite**
