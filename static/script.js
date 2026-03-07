// script.js — WellBeing Agent Frontend
// Handles chat, voice recording, language detection, and UI state

// ═══════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════
let isGenerating = false;   // blocks input while waiting for LLM
let messageCount = 0;
let currentLanguage = "english";
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

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
// Send Message  (blocks UI while generating)
// ═══════════════════════════════════════════════════════════════════════════
async function sendMessage(overrideText = null) {
    if (isGenerating) return;          // ← block duplicate sends

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
        setGenerating(false);          // ← always unlock
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
                    `<span class="source-tag"><i class="fas fa-bookmark"></i> ${s.topic || "Source"}${
                        s.source ? " — " + s.source : ""
                    }</span>`
            )
            .join("");
        sourcesHtml = `<div class="sources-container"><div class="sources-label"><i class="fas fa-book-medical"></i> Sources</div><div class="sources-list">${items}</div></div>`;
    }

    // Set direction on the content paragraph, not the entire message div
    // This keeps the avatar/bubble layout correct while making text RTL
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
    // Set chat language based on which tab the question came from
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

let pendingVoiceBlob = null;

async function startRecording() {
    try {
        // Remove any existing voice preview
        removeVoicePreview();

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach((t) => t.stop());
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            pendingVoiceBlob = blob;
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

// ── Voice Preview ────────────────────────────────────────────────────────
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

    const sendBtn = document.createElement("button");
    sendBtn.className = "voice-preview-btn send-voice";
    sendBtn.title = "Send";
    sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send';

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "voice-preview-btn cancel-voice";
    cancelBtn.title = "Cancel";
    cancelBtn.innerHTML = '<i class="fas fa-times"></i>';

    reRecordBtn.addEventListener("click", () => {
        URL.revokeObjectURL(url);
        preview.remove();
        pendingVoiceBlob = null;
        startRecording();
    });

    sendBtn.addEventListener("click", () => {
        const blobToSend = pendingVoiceBlob;
        URL.revokeObjectURL(url);
        preview.remove();
        pendingVoiceBlob = null;
        if (blobToSend) sendVoice(blobToSend);
    });

    cancelBtn.addEventListener("click", () => {
        URL.revokeObjectURL(url);
        preview.remove();
        pendingVoiceBlob = null;
    });

    actions.appendChild(reRecordBtn);
    actions.appendChild(sendBtn);
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
    pendingVoiceBlob = null;
}

function reRecord() {
    removeVoicePreview();
    startRecording();
}

// ═══════════════════════════════════════════════════════════════════════════
// ✅ FIXED: sendVoice with FileReader error handling and fetch timeout
// ═══════════════════════════════════════════════════════════════════════════
async function sendVoice(blob) {
    setGenerating(true);
    showTypingIndicator();
    try {
        // Convert blob to base64 with error handling
        const base64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(",")[1]);
            reader.onerror = (err) => reject(new Error('FileReader error: ' + err));
            reader.readAsDataURL(blob);
        });

        // Fetch with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const resp = await fetch("/voice-query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64 }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        hideTypingIndicator();

        if (!resp.ok) {
            const errorText = await resp.text();
            throw new Error(`Server error ${resp.status}: ${errorText}`);
        }

        const data = await resp.json();

        // Display transcribed text if available
        if (data.transcribed_text) {
            addMessageToChat(data.transcribed_text, "user", data.language || "english");
        }
        // Always display the answer
        addMessageToChat(
            data.answer,
            "system",
            data.language || "english",
            data.sources || []
        );
    } catch (err) {
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

// ═════════════════════════════════════════════════════════════════════════
// Language Detection (client-side quick check with Roman Urdu support)
// ═════════════════════════════════════════════════════════════════════════

// Common Roman Urdu words for client-side detection
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
    // 1) Check for Urdu script
    const urduRange = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/;
    if (urduRange.test(text)) return "urdu";

    // 2) Check for Roman Urdu
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

    // Strip foreign characters that LLMs sometimes leak into Urdu responses
    text = text.replace(/[\u0900-\u097F]/g, "");           // Devanagari (Hindi)
    text = text.replace(/[\u4E00-\u9FFF]/g, "");           // CJK (Chinese)
    text = text.replace(/[\u1E00-\u1EFF]/g, "");           // Latin Extended Additional (Vietnamese)
    text = text.replace(/[\u0300-\u036F]/g, "");           // Combining diacritical marks
    // NOTE: Do NOT strip Latin words — Urdu responses legitimately use
    // English medical terms like cancer, chemo, DNA, etc.

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