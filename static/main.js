// ==================================================
// 5. HISTORY LOGIC
// ==================================================

// Select Elements
const historyBtn = document.getElementById('historyBtn');
const historyPanel = document.getElementById('historyPanel');
const closeHistory = document.getElementById('closeHistory');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

// Load history when page starts
renderHistory();

// Toggle Panel
historyBtn.addEventListener('click', () => {
    historyPanel.classList.remove('hidden');
});

closeHistory.addEventListener('click', () => {
    historyPanel.classList.add('hidden');
});

// Clear History
clearHistoryBtn.addEventListener('click', () => {
    localStorage.removeItem('pigHistory');
    renderHistory();
});

// Function to Save a New Log
function saveToHistory(text, id) {
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    const entry = {
        text: text,
        id: id,
        time: timeString,
        type: text.includes("Background Noise") ? "healthy" : "disease"
    };

    // Get existing history
    const history = JSON.parse(localStorage.getItem('pigHistory') || '[]');
    
    // Add new entry to the TOP
    history.unshift(entry);
    
    // Limit to last 20 items (optional)
    if (history.length > 20) history.pop();

    // Save back
    localStorage.setItem('pigHistory', JSON.stringify(history));
    
    // Refresh UI
    renderHistory();
}

// Function to Render the List
function renderHistory() {
    const history = JSON.parse(localStorage.getItem('pigHistory') || '[]');
    const historyList = document.getElementById('historyList');
    const badge = document.getElementById('historyCount'); // <--- Get the badge

    // 1. UPDATE THE BADGE COUNT
    const count = history.length;
    badge.textContent = count;

    if (count > 0) {
        badge.classList.remove('hidden');
    } else {
        badge.classList.add('hidden');
    }

    // 2. RENDER THE LIST (Standard logic)
    historyList.innerHTML = ""; 

    if (history.length === 0) {
        historyList.innerHTML = '<li class="empty-msg" style="color:#999; text-align:center; padding:10px;">No recent alerts.</li>';
        return;
    }

    history.forEach(item => {
        const li = document.createElement('li');
        li.className = `history-item ${item.type}`;
        li.innerHTML = `
            <strong>Pig ${item.id}:</strong> ${item.text}
            <span class="timestamp">${item.time}</span>
        `;
        historyList.appendChild(li);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    console.log("Page loaded. Resetting Audio/Bars only...");
    fetch('/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: 'reset_audio' }) // <--- CHANGED THIS
    });
    
    stopPolling();
    // ==================================================
    // 1. NAVIGATION & UI SETUP
    // ==================================================
    const loadingScreen = document.querySelector('.loading-screen');
    const homepage = document.querySelector('.homepage');
    const monitoringPage = document.querySelector('.monitoring-page');
    
    const monitorBtn = document.querySelector('.monitor'); // The big SVG button
    const backBtn = document.getElementById('backBtn'); 
    
    // Handle Loading Screen Animation
    loadingScreen.addEventListener('animationend', () => {
        loadingScreen.classList.add('hidden');
        homepage.classList.remove('hidden');
    });

    // Go to Live Monitor
    monitorBtn.addEventListener('click', () => {
        homepage.classList.add('hidden');
        monitoringPage.classList.remove('hidden');
    });

    // Go Back to Home
    backBtn.addEventListener('click', () => {
        monitoringPage.classList.add('hidden');
        homepage.classList.remove('hidden');
        // Optional: Send reset command when leaving page
        stopPolling();
        fetch('/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'reset' })
        });
    });
});

// ==================================================
// 2. OPTIMIZED POLLING LOGIC
// ==================================================
let pollingInterval = null;

function startPolling() {
    // Clear any existing poller to prevent duplicates
    if (pollingInterval) clearInterval(pollingInterval);

    console.log("Started polling for AI results...");

    // Ask the server for status every 500ms
    pollingInterval = setInterval(() => {
        fetch('/get_status')
            .then(response => response.json())
            .then(data => {
                // Check if analysis is complete (timestamp > 0)
                if (data.timestamp > 0) {
                    
                    // SAFETY CHECK: Ignore if text is still placeholder
                    if (data.text === "Analyzing..." || data.text === "System Ready") {
                        return; 
                    }

                    console.log("Result received:", data.text);
                    
                    // Show the notification with the Result Text and Pig ID
                    showNotification(data.text, data.id);

                    // STOP POLLING (Save resources)
                    stopPolling();
                }
            })
            .catch(err => {
                console.error("Polling error:", err);
                stopPolling();
            });
    }, 500);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log("Stopped polling.");
    }
}

// ==================================================
// 3. KEYBOARD CONTROLS
// ==================================================
document.addEventListener('keydown', (e) => {
    // Only allow controls if we are on the monitoring page
    const monitoringPage = document.querySelector('.monitoring-page');
    if (monitoringPage.classList.contains('hidden')) return;

    // A. Number Keys (0-9) -> Start Analysis
    if (e.key >= '0' && e.key <= '9') {
        const id = e.key;
        console.log(`Selected Pig ID: ${id}`);

        // 1. Send command to Python
        fetch('/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'select_id', value: id })
        });

        // 2. Start waiting for the result
        startPolling();
    }
    
    // B. 'N' Key -> Reset / Stop
    if (e.key === 'n' || e.key === 'N') {
        console.log("Resetting system...");
        fetch('/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command: 'reset' })
        });

        stopPolling();
    }
});

// ==================================================
// 4. NOTIFICATION UI LOGIC
// ==================================================
function showNotification(predictionText, pigId) {
    saveToHistory(predictionText, pigId);

    const notif = document.getElementById('ai-notification');
    const title = document.getElementById('notif-title');
    const message = document.getElementById('notif-message');

    // Set the body text (e.g., "PRRS (98.5%)")
    message.textContent = predictionText;
    
    // Reset classes to ensure animation replays if needed
    notif.className = "notification show"; 

    // Dynamic Title & Color based on result
    if (predictionText.includes("Background Noise")) {
        // HEALTHY CASE
        title.textContent = `Pig ${pigId} is Healthy`; 
        notif.classList.add("success"); // Green border
    } else {
        // DISEASE CASE
        title.textContent = `Pig ${pigId}: Disease Detected`;
        notif.classList.add("danger");  // Red border
    }

    // Hide automatically after 6 seconds
    setTimeout(() => {
        notif.classList.remove("show");
    }, 6000);
}
