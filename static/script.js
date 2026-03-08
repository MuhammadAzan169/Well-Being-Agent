// script.js — WellBeing Agent Frontend
// Handles chat, voice recording, language detection, and UI state

// ═══════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════
let isGenerating = false;
let messageCount = 0;
let currentLanguage = "english";
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let pendingVoiceBlob = null;          // recorded audio waiting to be sent

// ═══════════════════════════════════════════════════════════════════════════
// DOM Ready
// ═══════════════════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    initChat();
    loadPredefinedQuestions("english");
    loadPredefinedQuestions("urdu");
    initTabs();
    initVoice();

    document.getElementById("welcomeTime").textContent = formatTime(new Date());
});

// ═══════════════════════════════════════════════════════════════════════════
// Chat Initialization
// ═══════════════════════════════════════════════════════════════════════════
function initChat() {
    const input = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendButton");

    sendBtn.addEventListener("click", () => sendMessage());
    input.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-detect Urdu script and switch input direction to RTL
    input.addEventListener("input", () => {
        const hasUrdu = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/.test(input.value);
        input.dir = hasUrdu ? "rtl" : "ltr";
        input.style.textAlign = hasUrdu ? "right" : "left";
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Send Message – handles both text and pending voice
// ═══════════════════════════════════════════════════════════════════════════
async function sendMessage(overrideText = null) {
    console.log("sendMessage called, pendingVoiceBlob =", pendingVoiceBlob ? "yes" : "no");
    if (isGenerating) {
        console.log("Already generating, ignoring click");
        return;
    }

    // If there's pending audio, send it
    if (pendingVoiceBlob) {
        const blob = pendingVoiceBlob;
        pendingVoiceBlob = null;          // clear immediately
        removeVoicePreview();              // hide preview
        document.getElementById("userInput").value = ""; // optional clear
        await sendVoice(blob);             // send audio
        return;
    }

    // Otherwise handle text input
    const input = document.getElementById("userInput");
    const message = overrideText || input.value.trim();
    if (!message) return;

    // Lock UI
    setGenerating(true);
    input.value = "";

    // Detect language
    currentLanguage = detectLanguage(message);
    updateLanguageDisplay(currentLanguage);

    // Show user bubble
    addMessageToChat(message, "user", currentLanguage);
    showTypingIndicator();

    try {
        const resp = await fetch("/ask-query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, language: currentLanguage }),
        });

        hideTypingIndicator();

        if (!resp.ok) {
            const errBody = await resp.json().catch(() => ({}));
            throw new Error(errBody.detail || `Server error ${resp.status}`);
        }

        const data = await resp.json();

        if (data.language) {
            currentLanguage = data.language;
            updateLanguageDisplay(currentLanguage);
        }

        addMessageToChat(
            data.answer,
            "system",
            data.language || currentLanguage,
            data.sources || []
        );
    } catch (err) {
        hideTypingIndicator();
        addMessageToChat(
            "Sorry, something went wrong. Please try again.",
            "system",
            "english"
        );
        console.error("Send error:", err);
    } finally {
        setGenerating(false);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UI Blocking helpers
// ═══════════════════════════════════════════════════════════════════════════
function setGenerating(state) {
    isGenerating = state;
    const input = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendButton");
    const voiceBtn = document.getElementById("voiceButton");

    input.disabled = state;
    sendBtn.disabled = state;
    if (voiceBtn) voiceBtn.disabled = state;

    if (state) {
        sendBtn.classList.add("disabled");
        input.placeholder = "Waiting for response…";
    } else {
        sendBtn.classList.remove("disabled");
        input.placeholder = "Ask about breast cancer support, treatment options, or recovery...";
        input.focus();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Render a chat message
// ═══════════════════════════════════════════════════════════════════════════
function addMessageToChat(message, sender, language = "english", sources = []) {
    const container = document.getElementById("chatMessages");
    const div = document.createElement("div");
    const isUrdu = language === "urdu";
    const isSystem = sender === "system";

    div.className = `message ${sender}-message${isUrdu ? " urdu-text" : ""}`;

    // Clean Urdu system responses (don't clean user input — they typed it)
    let displayMsg = (isUrdu && isSystem) ? cleanUrduText(message) : message;

    // Build sources HTML
    let sourcesHtml = "";
    if (isSystem && sources.length > 0) {
        const items = sources
            .slice(0, 3)
            .map(
                (s) =>
                    `<span class="source-tag"><i class="fas fa-bookmark"></i> ${s.topic || "Source"}${s.source ? " — " + s.source : ""}</span>`
            )
            .join("");
        sourcesHtml = `<div class="sources-container"><div class="sources-label"><i class="fas fa-book-medical"></i> Sources</div><div class="sources-list">${items}</div></div>`;
    }

    // Set direction on the content paragraph
    const contentDir = isUrdu ? ' dir="rtl"' : '';

    div.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${isSystem ? "robot" : "user"}"></i>
        </div>
        <div class="message-content${isUrdu ? " urdu-content" : ""}"${contentDir}>
            <p>${formatMessage(displayMsg)}</p>
            ${sourcesHtml}
            <span class="message-time">${formatTime(new Date())}</span>
        </div>`;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;

    messageCount++;
    const countEl = document.getElementById("messageCount");
    if (countEl) countEl.textContent = messageCount;
}

// ═══════════════════════════════════════════════════════════════════════════
// Voice Recording
// ═══════════════════════════════════════════════════════════════════════════
function initVoice() {
    const btn = document.getElementById("voiceButton");
    if (!btn) return;
    btn.addEventListener("click", toggleRecording);
}

async function toggleRecording() {
    if (isGenerating) return;
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        removeVoicePreview(); // remove any existing preview

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach((t) => t.stop());
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            pendingVoiceBlob = blob;               // store for later sending
            showVoicePreview(blob);
        };

        mediaRecorder.start();
        isRecording = true;
        updateVoiceUI(true);
    } catch (e) {
        console.error("Mic error:", e);
        alert("Could not access microphone. Please check permissions.");
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
    isRecording = false;
    updateVoiceUI(false);
}

function updateVoiceUI(recording) {
    const btn = document.getElementById("voiceButton");
    if (!btn) return;
    btn.classList.toggle("recording", recording);
    btn.innerHTML = recording
        ? '<i class="fas fa-stop"></i>'
        : '<i class="fas fa-microphone"></i>';
}

// ── Voice Preview (Re-record & Cancel only; Send uses main send button) ──
function showVoicePreview(blob) {
    removeVoicePreview();
    const url = URL.createObjectURL(blob);

    const preview = document.createElement("div");
    preview.id = "voicePreview";
    preview.className = "voice-preview";

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = url;

    const actions = document.createElement("div");
    actions.className = "voice-preview-actions";

    const reRecordBtn = document.createElement("button");
    reRecordBtn.className = "voice-preview-btn re-record";
    reRecordBtn.title = "Re-record";
    reRecordBtn.innerHTML = '<i class="fas fa-redo"></i> Re-record';

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "voice-preview-btn cancel-voice";
    cancelBtn.title = "Cancel";
    cancelBtn.innerHTML = '<i class="fas fa-times"></i> Cancel';

    reRecordBtn.addEventListener("click", () => {
        URL.revokeObjectURL(url);
        preview.remove();
        pendingVoiceBlob = null;
        startRecording();
    });

    cancelBtn.addEventListener("click", () => {
        URL.revokeObjectURL(url);
        preview.remove();
        pendingVoiceBlob = null;
    });

    actions.appendChild(reRecordBtn);
    actions.appendChild(cancelBtn);
    preview.appendChild(audio);
    preview.appendChild(actions);

    const inputContainer = document.querySelector(".chat-input-container");
    inputContainer.insertBefore(preview, inputContainer.firstChild);
}

function removeVoicePreview() {
    const existing = document.getElementById("voicePreview");
    if (existing) {
        const audio = existing.querySelector("audio");
        if (audio && audio.src) URL.revokeObjectURL(audio.src);
        existing.remove();
    }
    // Do NOT clear pendingVoiceBlob here – it's cleared when send or re-record
}

// ═══════════════════════════════════════════════════════════════════════════
// ✅ FIXED: sendVoice – sends blob to backend and displays transcription + answer
// ═══════════════════════════════════════════════════════════════════════════
async function sendVoice(blob) {
    console.log("sendVoice: starting, blob size:", blob.size);
    setGenerating(true);
    showTypingIndicator();

    // Safety timeout: if something hangs, reset UI after 35 seconds
    const safetyTimeout = setTimeout(() => {
        console.warn("sendVoice safety timeout triggered");
        hideTypingIndicator();
        setGenerating(false);
        addMessageToChat("Request timed out. Please try again.", "system", "english");
    }, 35000);

    try {
        // Convert blob to base64
        const base64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(",")[1]);
            reader.onerror = (err) => reject(new Error('FileReader error: ' + err));
            reader.readAsDataURL(blob);
        });

        console.log("sendVoice: base64 conversion done, length:", base64.length);

        const controller = new AbortController();
        const fetchTimeout = setTimeout(() => controller.abort(), 30000);

        const resp = await fetch("/voice-query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64 }),
            signal: controller.signal
        });
        clearTimeout(fetchTimeout);
        clearTimeout(safetyTimeout);  // clear safety if fetch returned

        hideTypingIndicator();

        console.log("sendVoice: response status", resp.status);

        if (!resp.ok) {
            const errorText = await resp.text();
            throw new Error(`Server error ${resp.status}: ${errorText}`);
        }

        const data = await resp.json();
        console.log("sendVoice: received data", data);

        // Display transcribed text as user message
        if (data.transcribed_text) {
            addMessageToChat(data.transcribed_text, "user", data.language || "english");
        } else {
            console.warn("No transcribed text in response");
        }

        // Display the answer
        addMessageToChat(
            data.answer,
            "system",
            data.language || "english",
            data.sources || []
        );
    } catch (err) {
        clearTimeout(safetyTimeout);
        hideTypingIndicator();
        let errorMsg = "Voice processing failed. Please type your question.";
        if (err.name === 'AbortError') {
            errorMsg = "Request timed out. Please try again.";
        } else if (err.message) {
            errorMsg = `Voice error: ${err.message}`;
        }
        addMessageToChat(errorMsg, "system", "english");
        console.error("Voice error:", err);
    } finally {
        setGenerating(false);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Predefined Questions
// ═══════════════════════════════════════════════════════════════════════════
async function loadPredefinedQuestions(lang) {
    try {
        const resp = await fetch(`/predefined-questions?language=${lang}`);
        if (!resp.ok) return;
        const data = await resp.json();
        const containerId = lang === "urdu" ? "urdu-questions" : "english-questions";
        const container = document.getElementById(containerId);
        if (!container || !data.questions) return;

        container.innerHTML = data.questions
            .map(
                (q) => `
            <div class="question-card ${lang === "urdu" ? "urdu-text" : ""}"
                 ${lang === "urdu" ? 'dir="rtl"' : ""}
                 onclick="askPredefined('${escapeHtml(q.question)}', '${lang}')">
                <i class="${q.icon || "fas fa-question-circle"}"></i>
                <span>${q.question}</span>
            </div>`
            )
            .join("");
    } catch (e) {
        console.warn(`Failed to load ${lang} questions:`, e);
    }
}

function askPredefined(question, lang) {
    if (isGenerating) return;
    if (lang) {
        currentLanguage = lang;
        updateLanguageDisplay(currentLanguage);
    }
    document.getElementById("userInput").value = question;
    sendMessage(question);
}

// ═══════════════════════════════════════════════════════════════════════════
// Tabs
// ═══════════════════════════════════════════════════════════════════════════
function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
            btn.classList.add("active");
            const tab = btn.dataset.tab;
            const content = document.getElementById(`${tab}-tab`);
            if (content) content.classList.add("active");
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Language Detection (client-side)
// ═══════════════════════════════════════════════════════════════════════════
const ROMAN_URDU_WORDS = new Set([
    "mera", "meri", "mere", "mujhe", "apna", "apni", "apne",
    "dard", "sar", "sir", "pet", "seena", "hath", "pair",
    "bohat", "bohot", "bahut", "kaise", "kya", "kyun", "kab",
    "hai", "hain", "tha", "thi", "raha", "rahi",
    "ilaj", "ilaaj", "dawa", "dawai", "doctor", "daktar",
    "cancer", "kenser", "chemo", "chemotherapy",
    "thakan", "kamzori", "bukhar", "ulti", "matli",
    "dar", "khauf", "fikar", "pareshani", "udasi",
    "batao", "batain", "chahiye", "sakta", "sakti",
    "ke baad", "ke doran", "ke liye", "ke sath",
    "acha", "achi", "theek", "nahi", "nahin", "haan",
    "doodh", "dudh", "bachcha", "bacche",
    "shukria", "shukriya", "meharbani",
]);

function detectLanguage(text) {
    if (/[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/.test(text)) return "urdu";
    const words = text.toLowerCase().split(/\s+/);
    let romanUrduCount = 0;
    for (const w of words) {
        if (ROMAN_URDU_WORDS.has(w)) romanUrduCount++;
    }
    if (romanUrduCount >= 2) return "urdu";
    return "english";
}

function updateLanguageDisplay(lang) {
    const el = document.getElementById("currentLanguageDisplay");
    if (el) el.textContent = lang === "urdu" ? "اردو" : "English";
    const stat = document.getElementById("activeLanguage");
    if (stat) stat.textContent = lang === "urdu" ? "اردو" : "English";
}

// ═══════════════════════════════════════════════════════════════════════════
// Urdu Text Cleaning
// ═══════════════════════════════════════════════════════════════════════════
function cleanUrduText(text) {
    if (!text) return text;
    const fixes = {
        "مجہے": "مجھے", "کہےنسر": "کینسر", "ڈڈاکٹر": "ڈاکٹر",
        "ہےہ": "ہے", "مہےں": "میں", "ہےں": "ہیں",
        "ھے": "ہے", "ھوں": "ہوں", "ھیں": "ہیں",
        "ےے": "ے", "ںں": "ں", "ہہ": "ہ",
        "ے لہےے": "کے لیے", "نہہےں": "نہیں",
        "بارے مہےں": "بارے میں", "کرہےں": "کریں",
        "برہےسٹ": "بریسٹ", "کہےموتھراپہے": "کیموتھراپی",
    };
    for (const [wrong, right] of Object.entries(fixes)) {
        text = text.replaceAll(wrong, right);
    }
    // Strip foreign characters
    text = text.replace(/[\u0900-\u097F]/g, "");           // Devanagari
    text = text.replace(/[\u4E00-\u9FFF]/g, "");           // CJK
    text = text.replace(/[\u1E00-\u1EFF]/g, "");           // Latin Extended Additional
    text = text.replace(/[\u0300-\u036F]/g, "");           // Combining diacritics
    return text.replace(/\s+/g, " ").trim();
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════
function showTypingIndicator() {
    const el = document.getElementById("typingIndicator");
    if (el) el.classList.add("visible");
}

function hideTypingIndicator() {
    const el = document.getElementById("typingIndicator");
    if (el) el.classList.remove("visible");
}

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatMessage(text) {
    if (!text) return "";
    return text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/\n/g, "<br>");
}

function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML.replace(/'/g, "\\'");
}