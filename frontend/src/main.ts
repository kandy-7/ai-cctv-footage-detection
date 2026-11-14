import './style.css'

const API_BASE = 'http://localhost:8001';

// ── Type definitions ──────────────────────────────────────────────────────────
interface EventResult {
  timestamp: number;
  video_url: string;
  classes: string[];
  count: number;
  confidence?: number;
}

interface QueryResponse {
  found: boolean;
  events: EventResult[];
  total_matches: number;
  message?: string;
}

interface StatusResponse {
  status: 'processing' | 'done';
  event_count: number;
}

interface AppearanceResult {
  video_id: string;
  filename: string | null;
  track_id: number | null;
  timestamp: number | null;
  frame_number: number | null;
  upper_color: string;
  upper_color_confidence: number;
  crop_url: string | null;
  clip_url: string | null;
}

interface AppearanceResponse {
  found: boolean;
  results: AppearanceResult[];
  total: number;
  query: string;
}

// ── App state ─────────────────────────────────────────────────────────────────
let currentVideoId: string | null = null;
let pollingTimer: ReturnType<typeof setInterval> | null = null;
let activeEventIndex = 0;

// ── HTML scaffold ─────────────────────────────────────────────────────────────
document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="container">
    <header class="header">
      <div class="logo-section">
        <svg class="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        <h1>AI CCTV Analysis</h1>
      </div>
      <div class="status-indicator" id="statusIndicator">
        <span class="status-dot"></span>
        <span id="statusText">System Ready</span>
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
          <p class="upload-text">Drag &amp; drop your video or <label for="fileInput" class="upload-link">browse</label></p>
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
          <p id="uploadSuccessText">Video uploaded! Processing...</p>
        </div>

        <!-- Processing Status -->
        <div class="processing-status" id="processingStatus" style="display: none;">
          <div class="processing-spinner"></div>
          <span id="processingText">Processing video... 0 events found</span>
        </div>
      </section>

      <!-- Query Section -->
      <section class="card query-section" id="querySection" style="display: none;">
        <div class="card-header">
          <h2>Natural Language Query</h2>
          <p>Ask questions about your footage in plain English</p>
        </div>

        <!-- Search Mode Tabs -->
        <div class="search-tabs">
          <button class="search-tab active" id="tabObject" data-tab="object">
            🔍 Object Search
          </button>
          <button class="search-tab" id="tabAppearance" data-tab="appearance">
            👕 Appearance Search
          </button>
        </div>

        <!-- Object Search Panel -->
        <div id="objectSearchPanel">
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
        </div>

        <!-- Appearance Search Panel -->
        <div id="appearanceSearchPanel" style="display:none;">
          <div class="query-input-container">
            <input
              type="text"
              id="appearanceInput"
              class="query-input"
              placeholder="e.g., 'orange T-shirt', 'blue jacket', 'black shirt'"
            />
            <button id="appearanceSearchBtn" class="btn btn-primary">
              <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Search
            </button>
          </div>
          <div class="query-examples">
            <p>Try these examples:</p>
            <div class="example-chips" id="appearanceChips">
              <button class="chip appearance-chip" data-query="orange shirt">🟠 Orange</button>
              <button class="chip appearance-chip" data-query="blue shirt">🔵 Blue</button>
              <button class="chip appearance-chip" data-query="black jacket">⚫ Black</button>
              <button class="chip appearance-chip" data-query="red shirt">🔴 Red</button>
              <button class="chip appearance-chip" data-query="white shirt">⚪ White</button>
              <button class="chip appearance-chip" data-query="pink shirt">🩷 Pink</button>
            </div>
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

      <!-- Appearance Results Section -->
      <section class="card results-section" id="appearanceResultsSection" style="display: none;">
        <div class="card-header">
          <h2>Appearance Search Results</h2>
        </div>
        <div id="appearanceResultsContent"></div>
      </section>
    </main>

    <footer class="footer">
      <p>Powered by YOLOv8 &amp; FastAPI | AI-Enhanced Video Analysis</p>
    </footer>
  </div>
`;

// ── DOM references ────────────────────────────────────────────────────────────
const uploadArea        = document.getElementById('uploadArea')        as HTMLDivElement;
const fileInput         = document.getElementById('fileInput')         as HTMLInputElement;
const progressContainer = document.getElementById('progressContainer') as HTMLDivElement;
const uploadSuccess     = document.getElementById('uploadSuccess')     as HTMLDivElement;
const uploadSuccessText = document.getElementById('uploadSuccessText') as HTMLParagraphElement;
const querySection      = document.getElementById('querySection')      as HTMLElement;
const processingStatus  = document.getElementById('processingStatus')  as HTMLDivElement;
const processingText    = document.getElementById('processingText')    as HTMLSpanElement;
const queryInput        = document.getElementById('queryInput')        as HTMLInputElement;
const searchBtn         = document.getElementById('searchBtn')         as HTMLButtonElement;
const resultsSection    = document.getElementById('resultsSection')    as HTMLElement;
const resultsContent    = document.getElementById('resultsContent')    as HTMLDivElement;
const statusText        = document.getElementById('statusText')        as HTMLSpanElement;

// Appearance DOM refs
const tabObject              = document.getElementById('tabObject')              as HTMLButtonElement;
const tabAppearance          = document.getElementById('tabAppearance')          as HTMLButtonElement;
const objectSearchPanel      = document.getElementById('objectSearchPanel')      as HTMLDivElement;
const appearanceSearchPanel  = document.getElementById('appearanceSearchPanel')  as HTMLDivElement;
const appearanceInput        = document.getElementById('appearanceInput')        as HTMLInputElement;
const appearanceSearchBtn    = document.getElementById('appearanceSearchBtn')    as HTMLButtonElement;
const appearanceResultsSection = document.getElementById('appearanceResultsSection') as HTMLElement;
const appearanceResultsContent = document.getElementById('appearanceResultsContent') as HTMLDivElement;

// ── Search tab switching ──────────────────────────────────────────────────────
function switchTab(tab: 'object' | 'appearance') {
  if (tab === 'object') {
    tabObject.classList.add('active');
    tabAppearance.classList.remove('active');
    objectSearchPanel.style.display = 'block';
    appearanceSearchPanel.style.display = 'none';
    appearanceResultsSection.style.display = 'none';
  } else {
    tabAppearance.classList.add('active');
    tabObject.classList.remove('active');
    appearanceSearchPanel.style.display = 'block';
    objectSearchPanel.style.display = 'none';
    resultsSection.style.display = 'none';
  }
}

tabObject.addEventListener('click', () => switchTab('object'));
tabAppearance.addEventListener('click', () => switchTab('appearance'));

// ── Upload ────────────────────────────────────────────────────────────────────
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
  if (files && files.length > 0) handleFileUpload(files[0]);
});

fileInput.addEventListener('change', (e) => {
  const files = (e.target as HTMLInputElement).files;
  if (files && files.length > 0) handleFileUpload(files[0]);
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
    uploadSuccessText.textContent = `Video uploaded! ID: ${currentVideoId}`;

    // Start polling
    startStatusPolling(currentVideoId!);

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
  processingStatus.style.display = 'none';
  querySection.style.display = 'none';
  if (pollingTimer !== null) {
    clearInterval(pollingTimer);
    pollingTimer = null;
  }
}

// ── Task A: Status polling ────────────────────────────────────────────────────
function startStatusPolling(videoId: string) {
  processingStatus.style.display = 'flex';
  processingText.textContent = 'Processing video... 0 events found';
  statusText.textContent = 'Processing...';

  async function poll() {
    try {
      const res = await fetch(`${API_BASE}/status/${videoId}`);
      if (!res.ok) return; // keep polling on transient errors
      const data: StatusResponse = await res.json();

      processingText.textContent = `Processing video... ${data.event_count} events found`;

      if (data.status === 'done') {
        stopPolling();
        processingText.textContent = `✓ Processing complete — ${data.event_count} events found`;
        statusText.textContent = 'Processing Complete';
        querySection.style.display = 'block';
        querySection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    } catch (err) {
      console.error('Status poll error:', err);
    }
  }

  poll(); // immediate first check
  pollingTimer = setInterval(poll, 3000);
}

function stopPolling() {
  if (pollingTimer !== null) {
    clearInterval(pollingTimer);
    pollingTimer = null;
  }
}

// ── Query / Search ────────────────────────────────────────────────────────────
searchBtn.addEventListener('click', handleSearch);
queryInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') handleSearch();
});

document.querySelectorAll('.chip:not(.appearance-chip)').forEach(chip => {
  chip.addEventListener('click', () => {
    const query = chip.getAttribute('data-query');
    if (query) {
      queryInput.value = query;
      handleSearch();
    }
  });
});

// Appearance chip clicks
document.querySelectorAll('.appearance-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const query = chip.getAttribute('data-query');
    if (query) {
      appearanceInput.value = query;
      handleAppearanceSearch();
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
        query,
        threshold: 0.5,
      }),
    });

    const data: QueryResponse = await response.json();
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

// ── Task B + C: Display results with layout + timeline ────────────────────────
function displayResults(data: QueryResponse, query: string) {
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
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return;
  }

  activeEventIndex = 0;
  const totalEvents = data.total_matches || data.events.length;
  const first = data.events[0];

  resultsContent.innerHTML = `
    <div class="result-found">
      <div class="result-header">
        <h3>Found ${totalEvents} matching moment${totalEvents !== 1 ? 's' : ''}</h3>
        <p class="query-text">Query: &ldquo;${query}&rdquo;</p>
      </div>

      <div class="results-layout">
        <!-- Left: event card list -->
        <div class="events-panel" id="eventsPanel">
          ${data.events.map((ev, i) => buildEventCard(ev, i)).join('')}
        </div>

        <!-- Right: video + timeline -->
        <div class="video-panel" id="videoPanel">
          <div class="video-container">
            <video
              controls
              autoplay
              class="result-video"
              id="mainVideo"
              src="${first.video_url}"
            >
              Your browser does not support the video tag.
            </video>
          </div>

          <!-- Timeline -->
          <div class="timeline-wrapper">
            <div class="timeline-container" id="timelineContainer">
              <div class="timeline-track" id="timelineTrack">
                <div class="timeline-progress" id="timelineProgress" style="width:0%"></div>
                <div class="timeline-playhead" id="timelinePlayhead" style="left:0%"></div>
              </div>
              <div id="timelineMarkers"></div>
            </div>
            <div class="timeline-labels">
              <span id="timeCurrent">0:00</span>
              <span id="timeDuration">--:--</span>
            </div>
          </div>

          <!-- Download -->
          <div class="result-actions" style="margin-top:1rem;">
            <a href="${first.video_url}" id="downloadLink" download class="btn btn-secondary">
              <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download Clip
            </a>
          </div>
        </div>
      </div>
    </div>
  `;

  // Wire up interactions after DOM update
  setTimeout(() => {
    wireResults(data.events);
  }, 0);

  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function buildEventCard(ev: EventResult, index: number): string {
  const confidenceVal = ev.confidence != null ? ev.confidence : ev.count / 10;
  const confPct = Math.min(100, Math.round(confidenceVal * 100));
  const confColor = confPct >= 70 ? '#4caf50' : confPct >= 40 ? '#ff9800' : '#f44336';

  return `
    <div class="event-card ${index === 0 ? 'active' : ''}" data-index="${index}" data-url="${ev.video_url}">
      <div class="event-card-inner">
        <div class="event-badge">${index + 1}</div>
        <div class="event-card-body">
          <div class="event-time">${formatTime(ev.timestamp)}</div>
          <div class="event-classes">${ev.classes.join(', ')}</div>
          <div class="event-conf-bar">
            <div class="event-conf-fill" style="width:${confPct}%;background:${confColor}"></div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function wireResults(events: EventResult[]) {
  const video        = document.getElementById('mainVideo')       as HTMLVideoElement;
  const downloadLink = document.getElementById('downloadLink')    as HTMLAnchorElement;
  const cards        = document.querySelectorAll<HTMLElement>('.event-card');
  const progress     = document.getElementById('timelineProgress') as HTMLDivElement;
  const playhead     = document.getElementById('timelinePlayhead') as HTMLDivElement;
  const timeCurrent  = document.getElementById('timeCurrent')    as HTMLSpanElement;
  const timeDuration = document.getElementById('timeDuration')    as HTMLSpanElement;
  const markersEl    = document.getElementById('timelineMarkers') as HTMLDivElement;

  // ── Event card clicks ──
  cards.forEach((card) => {
    card.addEventListener('click', () => {
      const idx = parseInt(card.getAttribute('data-index') || '0', 10);
      selectEvent(idx, events, cards, video, downloadLink, markersEl);
    });
  });

  // ── Video metadata / time update for timeline ──
  video.addEventListener('loadedmetadata', () => {
    const dur = video.duration;
    timeDuration.textContent = isFinite(dur) ? formatTime(dur) : '--:--';
    renderTimelineMarkers(events, markersEl, dur, (idx) => {
      selectEvent(idx, events, cards, video, downloadLink, markersEl);
    });
  });

  video.addEventListener('timeupdate', () => {
    const dur = video.duration;
    const cur = video.currentTime;
    if (!isFinite(dur) || dur === 0) return;
    const pct = (cur / dur) * 100;
    progress.style.width = `${pct}%`;
    playhead.style.left  = `${pct}%`;
    timeCurrent.textContent = formatTime(cur);
  });

  // Initial marker render if metadata already loaded
  if (isFinite(video.duration) && video.duration > 0) {
    timeDuration.textContent = formatTime(video.duration);
    renderTimelineMarkers(events, markersEl, video.duration, (idx) => {
      selectEvent(idx, events, cards, video, downloadLink, markersEl);
    });
  }
}

function selectEvent(
  idx: number,
  events: EventResult[],
  cards: NodeListOf<HTMLElement>,
  video: HTMLVideoElement,
  downloadLink: HTMLAnchorElement,
  markersEl: HTMLDivElement,
) {
  activeEventIndex = idx;
  const ev = events[idx];

  // Update active card
  cards.forEach((c) => c.classList.remove('active'));
  cards[idx]?.classList.add('active');
  cards[idx]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Load clip
  video.src = ev.video_url;
  video.load();
  video.currentTime = 0;
  video.play().catch(() => {/* autoplay policy */ });

  // Update download
  downloadLink.href = ev.video_url;

  // Reset timeline progress
  const progress  = document.getElementById('timelineProgress') as HTMLDivElement;
  const playhead  = document.getElementById('timelinePlayhead') as HTMLDivElement;
  const timeCurrent = document.getElementById('timeCurrent')    as HTMLSpanElement;
  const timeDuration = document.getElementById('timeDuration')  as HTMLSpanElement;
  if (progress)  progress.style.width = '0%';
  if (playhead)  playhead.style.left  = '0%';
  if (timeCurrent) timeCurrent.textContent = '0:00';
  if (timeDuration) timeDuration.textContent = '--:--';

  // Highlight active marker
  document.querySelectorAll<HTMLElement>('.timeline-marker').forEach((m, i) => {
    m.classList.toggle('active-marker', i === idx);
  });

  // Re-render markers once new clip loads
  video.addEventListener('loadedmetadata', () => {
    const dur = video.duration;
    if (timeDuration) timeDuration.textContent = isFinite(dur) ? formatTime(dur) : '--:--';
    renderTimelineMarkers(events, markersEl, dur, (i) => {
      selectEvent(i, events, cards, video, downloadLink, markersEl);
    });
  }, { once: true });
}

// ── Task C: Timeline markers ──────────────────────────────────────────────────
function renderTimelineMarkers(
  events: EventResult[],
  container: HTMLDivElement,
  totalDuration: number,
  onClick: (idx: number) => void,
) {
  container.innerHTML = '';
  if (!isFinite(totalDuration) || totalDuration <= 0) return;

  events.forEach((ev, i) => {
    const pct = Math.min(99, Math.max(1, (ev.timestamp / totalDuration) * 100));
    const marker = document.createElement('div');
    marker.className = `timeline-marker${i === activeEventIndex ? ' active-marker' : ''}`;
    marker.style.left = `${pct}%`;
    marker.title = `Event ${i + 1}: ${formatTime(ev.timestamp)} — ${ev.classes.join(', ')}`;
    marker.textContent = '▲';
    marker.addEventListener('click', () => onClick(i));
    container.appendChild(marker);
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ── Appearance search handlers ────────────────────────────────────────────────
appearanceSearchBtn.addEventListener('click', handleAppearanceSearch);
appearanceInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') handleAppearanceSearch();
});

async function handleAppearanceSearch() {
  const query = appearanceInput.value.trim();
  if (!query) return;

  appearanceSearchBtn.disabled = true;
  appearanceSearchBtn.innerHTML = `
    <svg class="btn-icon spinning" viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
    Searching...
  `;

  try {
    const body: Record<string, unknown> = { query, confidence_min: 0.3, limit: 30 };
    if (currentVideoId) body.video_id = currentVideoId;

    const response = await fetch(`${API_BASE}/appearance-search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data: AppearanceResponse = await response.json();
    displayAppearanceResults(data);

  } catch (error) {
    console.error('Appearance search error:', error);
    alert('Appearance search failed. Please try again.');
  } finally {
    appearanceSearchBtn.disabled = false;
    appearanceSearchBtn.innerHTML = `
      <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      Search
    `;
  }
}

const COLOR_HEX: Record<string, string> = {
  red: '#e53935', orange: '#ff6b35', yellow: '#fdd835',
  green: '#43a047', blue: '#1e88e5', purple: '#8e24aa',
  pink: '#e91e63', white: '#f5f5f5', black: '#212121',
  other: '#9e9e9e', unknown: '#757575',
};

function displayAppearanceResults(data: AppearanceResponse) {
  appearanceResultsSection.style.display = 'block';

  if (!data.found || !data.results || data.results.length === 0) {
    appearanceResultsContent.innerHTML = `
      <div class="no-results">
        <svg class="no-results-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3>No Matching Appearance Found</h3>
        <p>No people matching that clothing colour were detected. Try a different colour or upload more footage.</p>
        <div class="tip-box">
          <p><strong>Note:</strong> Appearance search only works on videos processed after this feature was enabled.</p>
        </div>
      </div>
    `;
    appearanceResultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    return;
  }

  appearanceResultsContent.innerHTML = `
    <div class="result-found">
      <div class="result-header">
        <h3>Found ${data.total} person appearance${data.total !== 1 ? 's' : ''}</h3>
        <p class="query-text">Query: &ldquo;${data.query}&rdquo;</p>
      </div>
      <div class="appearance-results-grid">
        ${data.results.map(r => buildAppearanceCard(r)).join('')}
      </div>
    </div>
  `;

  // Wire "View Clip" buttons
  document.querySelectorAll<HTMLButtonElement>('.appearance-clip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const url = btn.getAttribute('data-clip-url');
      if (!url) return;
      // Show in a simple overlay or switch to object results video panel
      const overlay = document.createElement('div');
      overlay.className = 'appearance-clip-overlay';
      overlay.innerHTML = `
        <div class="appearance-clip-modal">
          <button class="appearance-clip-close" id="closeClipModal">✕ Close</button>
          <video controls autoplay class="result-video" src="${url}">
            Your browser does not support video.
          </video>
        </div>
      `;
      document.body.appendChild(overlay);
      document.getElementById('closeClipModal')?.addEventListener('click', () => {
        document.body.removeChild(overlay);
      });
    });
  });

  appearanceResultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function buildAppearanceCard(r: AppearanceResult): string {
  const colorHex = COLOR_HEX[r.upper_color] ?? '#9e9e9e';
  const confPct  = Math.round((r.upper_color_confidence ?? 0) * 100);
  const timeStr  = r.timestamp != null ? formatTime(r.timestamp) : '--';
  const filename = r.filename ?? r.video_id;

  const cropHtml = r.crop_url
    ? `<img src="${r.crop_url}" alt="Person crop" class="appearance-crop-img" onerror="this.style.display='none'" />`
    : `<div class="appearance-crop-placeholder">No crop</div>`;

  const clipBtn = r.clip_url
    ? `<button class="btn btn-secondary appearance-clip-btn" data-clip-url="${r.clip_url}" style="font-size:0.75rem;padding:0.35rem 0.8rem;">▶ View Clip</button>`
    : `<span class="appearance-no-clip">No clip yet</span>`;

  return `
    <div class="appearance-card">
      ${cropHtml}
      <div class="appearance-card-body">
        <div class="color-badge">
          <span class="color-dot" style="background:${colorHex}"></span>
          <span class="color-label">${r.upper_color}</span>
          <span class="color-conf">${confPct}%</span>
        </div>
        <div class="appearance-meta">
          <span>⏱ ${timeStr}</span>
          ${r.track_id != null ? `<span>🏷 Track ${r.track_id}</span>` : ''}
        </div>
        <div class="appearance-filename" title="${filename}">${filename}</div>
        <div class="appearance-actions">${clipBtn}</div>
      </div>
    </div>
  `;
}
