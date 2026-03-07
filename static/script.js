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
    if (isUrdu) div.setAttribute("dir", "rtl");

    // Clean Urdu if needed
    let displayMsg = isUrdu ? cleanUrduText(message) : message;

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

    div.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${isSystem ? "robot" : "user"}"></i>
        </div>
        <div class="message-content${isUrdu ? " urdu-content" : ""}">
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
                 onclick="askPredefined('${escapeHtml(q.question)}')">
                <i class="${q.icon || "fas fa-question-circle"}"></i>
                <span>${q.question}</span>
            </div>`
            )
            .join("");
    } catch (e) {
        console.warn(`Failed to load ${lang} questions:`, e);
    }
}

function askPredefined(question) {
    if (isGenerating) return;
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

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach((t) => t.stop());
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            await sendVoice(blob);
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

async function sendVoice(blob) {
    setGenerating(true);
    showTypingIndicator();
    try {
        const reader = new FileReader();
        const base64 = await new Promise((res) => {
            reader.onloadend = () => res(reader.result.split(",")[1]);
            reader.readAsDataURL(blob);
        });

        const resp = await fetch("/voice-query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64 }),
        });

        hideTypingIndicator();

        if (!resp.ok) throw new Error(`Voice error ${resp.status}`);
        const data = await resp.json();

        if (data.transcribed_text) {
            addMessageToChat(data.transcribed_text, "user", data.language || "english");
        }
        addMessageToChat(
            data.answer,
            "system",
            data.language || "english",
            data.sources || []
        );
    } catch (e) {
        hideTypingIndicator();
        addMessageToChat("Voice processing failed. Please type your question.", "system", "english");
        console.error("Voice error:", e);
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
