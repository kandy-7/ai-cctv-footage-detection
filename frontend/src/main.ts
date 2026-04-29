import './style.css'

const API_BASE = 'http://localhost:8001';

let currentVideoId: string | null = null;

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="container">
    <header class="header">
      <div class="logo-section">
        <svg class="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        <h1>AI CCTV Analysis</h1>
      </div>
      <div class="status-indicator">
        <span class="status-dot"></span>
        <span>System Ready</span>
      </div>
    </header>

    <main class="main-content">
      <!-- Upload Section -->
      <section class="card upload-section">
        <div class="card-header">
          <h2>Upload CCTV Footage</h2>
          <p>Upload your video file for AI-powered analysis</p>
        </div>
        <div class="upload-area" id="uploadArea">
          <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p class="upload-text">Drag & drop your video or <label for="fileInput" class="upload-link">browse</label></p>
          <p class="upload-hint">Supports MP4, AVI, MOV formats</p>
          <input type="file" id="fileInput" accept="video/*" hidden />
        </div>
        <div class="progress-container" id="progressContainer" style="display: none;">
          <div class="progress-bar">
            <div class="progress-fill" id="progressFill"></div>
          </div>
          <p class="progress-text" id="progressText">Uploading...</p>
        </div>
        <div class="upload-success" id="uploadSuccess" style="display: none;">
          <svg class="success-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          <p>Video uploaded successfully! Processing in progress...</p>
        </div>
      </section>

      <!-- Query Section -->
      <section class="card query-section" id="querySection" style="display: none;">
        <div class="card-header">
          <h2>Natural Language Query</h2>
          <p>Ask questions about your footage in plain English</p>
        </div>
        <div class="query-input-container">
          <input 
            type="text" 
            id="queryInput" 
            class="query-input" 
            placeholder="e.g., 'Show me when a person appears' or 'Find cars in the video'"
          />
          <button id="searchBtn" class="btn btn-primary">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Search
          </button>
        </div>
        <div class="query-examples">
          <p>Try these examples:</p>
          <div class="example-chips">
            <button class="chip" data-query="person walking">Person walking</button>
            <button class="chip" data-query="car driving">Car driving</button>
            <button class="chip" data-query="show me cats">Show cats</button>
            <button class="chip" data-query="person with backpack">Person with backpack</button>
          </div>
        </div>
      </section>

      <!-- Results Section -->
      <section class="card results-section" id="resultsSection" style="display: none;">
        <div class="card-header">
          <h2>Search Results</h2>
        </div>
        <div id="resultsContent"></div>
      </section>
    </main>

    <footer class="footer">
      <p>Powered by YOLOv8 & FastAPI | AI-Enhanced Video Analysis</p>
    </footer>
  </div>
`;

// Upload functionality
const uploadArea = document.getElementById('uploadArea') as HTMLDivElement;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const progressContainer = document.getElementById('progressContainer') as HTMLDivElement;
const uploadSuccess = document.getElementById('uploadSuccess') as HTMLDivElement;
const querySection = document.getElementById('querySection') as HTMLElement;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    handleFileUpload(files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  const files = (e.target as HTMLInputElement).files;
  if (files && files.length > 0) {
    handleFileUpload(files[0]);
  }
});

async function handleFileUpload(file: File) {
  uploadArea.style.display = 'none';
  progressContainer.style.display = 'block';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error('Upload failed');

    const data = await response.json();
    currentVideoId = data.video_id;

    progressContainer.style.display = 'none';
    uploadSuccess.style.display = 'flex';
    querySection.style.display = 'block';

    // Simulate processing time
    setTimeout(() => {
      const statusText = document.querySelector('.status-indicator span:last-child') as HTMLSpanElement;
      statusText.textContent = 'Processing Complete';
    }, 2000);

  } catch (error) {
    console.error('Upload error:', error);
    alert('Failed to upload video. Please try again.');
    resetUpload();
  }
}

function resetUpload() {
  uploadArea.style.display = 'flex';
  progressContainer.style.display = 'none';
  uploadSuccess.style.display = 'none';
}

// Query functionality
const queryInput = document.getElementById('queryInput') as HTMLInputElement;
const searchBtn = document.getElementById('searchBtn') as HTMLButtonElement;
const resultsSection = document.getElementById('resultsSection') as HTMLElement;
const resultsContent = document.getElementById('resultsContent') as HTMLDivElement;

searchBtn.addEventListener('click', handleSearch);
queryInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') handleSearch();
});

// Example chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const query = chip.getAttribute('data-query');
    if (query) {
      queryInput.value = query;
      handleSearch();
    }
  });
});

async function handleSearch() {
  const query = queryInput.value.trim();
  if (!query || !currentVideoId) return;

  searchBtn.disabled = true;
  searchBtn.innerHTML = `
    <svg class="btn-icon spinning" viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
    Searching...
  `;

  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_id: currentVideoId,
        query: query,
        threshold: 0.5
      })
    });

    const data = await response.json();
    displayResults(data, query);

  } catch (error) {
    console.error('Search error:', error);
    alert('Search failed. Please try again.');
  } finally {
    searchBtn.disabled = false;
    searchBtn.innerHTML = `
      <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      Search
    `;
  }
}

function displayResults(data: any, query: string) {
  resultsSection.style.display = 'block';

  if (!data.found || !data.events || data.events.length === 0) {
    resultsContent.innerHTML = `
      <div class="no-results">
        <svg class="no-results-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3>No Results Found</h3>
        <p>${data.message || 'The AI could not find any matching events.'}</p>
        <div class="tip-box">
          <p><strong>Tip:</strong> Try searching for "accident", "crash", or specific objects like "car and bike".</p>
        </div>
      </div>
    `;
  } else {
    const totalEvents = data.total_matches || data.events.length;
    const firstEvent = data.events[0];

    resultsContent.innerHTML = `
      <div class="result-found">
        <div class="result-header">
          <div class="result-info">
            <h3>Found ${totalEvents} matching moments</h3>
            <p>The AI identified multiple events. Select a timestamp below to view the clip.</p>
            <p class="query-text">Query: "${query}"</p>
          </div>
        </div>
        
        <div class="event-selector">
          <p>Detected Events:</p>
          <div class="event-timeline" id="eventTimeline">
            ${data.events.map((ev: any, index: number) => `
              <button class="event-chip ${index === 0 ? 'active' : ''}" 
                      data-url="${ev.video_url}" 
                      data-timestamp="${ev.timestamp}"
                      data-classes="${ev.classes.join(', ')}">
                <span class="time">${formatTime(ev.timestamp)}</span>
                <span class="desc">${ev.classes.slice(0, 2).join(', ')}${ev.classes.length > 2 ? '...' : ''}</span>
              </button>
            `).join('')}
          </div>
        </div>

        <div class="video-display-area" id="videoDisplayArea">
          <div class="video-container">
            <video controls autoplay class="result-video" id="mainVideo">
              <source src="${firstEvent.video_url}" type="video/mp4">
              Your browser does not support the video tag.
            </video>
            <div class="video-badge">4K Enhanced</div>
          </div>
          <div class="event-details" id="eventDetails">
             <p>Showing event at <strong>${formatTime(firstEvent.timestamp)}</strong></p>
             <p>Detected: ${firstEvent.classes.join(', ')}</p>
          </div>
          <div class="result-actions">
            <a href="${firstEvent.video_url}" id="downloadLink" download class="btn btn-secondary">
              <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download Clip
            </a>
          </div>
        </div>
      </div>
    `;

    // Add event listeners to chips
    setTimeout(() => {
      const chips = document.querySelectorAll('.event-chip');
      const video = document.getElementById('mainVideo') as HTMLVideoElement;
      const downloadLink = document.getElementById('downloadLink') as HTMLAnchorElement;
      const details = document.getElementById('eventDetails') as HTMLDivElement;

      chips.forEach(chip => {
        chip.addEventListener('click', () => {
          // Update active state
          chips.forEach(c => c.classList.remove('active'));
          chip.classList.add('active');

          // Update video source
          const url = chip.getAttribute('data-url')!;
          const timestamp = chip.getAttribute('data-timestamp')!;
          const classes = chip.getAttribute('data-classes')!;

          video.src = url;
          video.load();
          video.play();

          // Update download link
          downloadLink.href = url;

          // Update details
          details.innerHTML = `
            <p>Showing event at <strong>${formatTime(parseFloat(timestamp))}</strong></p>
            <p>Detected: ${classes}</p>
          `;
        });
      });
    }, 0);
  }

  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
