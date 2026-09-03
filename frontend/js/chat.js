// chat.js — WellBeing Agent chat app
// Handles chat, voice recording, language detection, UI state, and the
// ambient particle canvas. All backend calls go through the api service.

import { api } from "./services/api.js";

// ── State ──────────────────────────────────────────────────────────────────
let isGenerating = false;
let messageCount = 0;
let currentLanguage = "english";
let isRecording = false;
// Separate from currentLanguage (which auto-updates from detected message
// text) so that explicitly picking Urdu for voice input doesn't get reset
// back to English after the next typed/transcribed message.
let voiceLang = "english";

// ── DOM ready ────────────────────────────────────────────────────────────--
document.addEventListener("DOMContentLoaded", () => {
  initChat();
  loadPredefinedQuestions("english");
  loadPredefinedQuestions("urdu");
  initTabs();
  initVoice();
  initParticleCanvas();
  const welcome = document.getElementById("welcomeTime");
  if (welcome) welcome.textContent = formatTime(new Date());
});

// ── Ambient particle canvas ─────────────────────────────────────────────--
function initParticleCanvas() {
  const canvas = document.getElementById("particleCanvas");
  if (!canvas) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const ctx = canvas.getContext("2d");
  let particles = [];
  const PARTICLE_COUNT = 20;
  const CONNECTION_DIST = 100;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  class Particle {
    constructor() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = (Math.random() - 0.5) * 0.4;
      this.radius = Math.random() * 2 + 1;
      this.opacity = Math.random() * 0.3 + 0.1;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, 107, 147, ${this.opacity})`;
      ctx.fill();
    }
  }

  for (let i = 0; i < PARTICLE_COUNT; i++) particles.push(new Particle());

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECTION_DIST) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(255, 107, 147, ${0.06 * (1 - dist / CONNECTION_DIST)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    particles.forEach((p) => {
      p.update();
      p.draw();
    });
    requestAnimationFrame(animate);
  }
  animate();
}

// ── Chat init ──────────────────────────────────────────────────────────--
function initChat() {
  const input = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendButton");

  sendBtn.addEventListener("click", (e) => {
    addRipple(e, sendBtn);
    sendMessage();
  });
  input.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener("input", () => {
    const hasUrdu = /[؀-ۿݐ-ݿࢠ-ࣿ]/.test(input.value);
    input.dir = hasUrdu ? "rtl" : "ltr";
    input.style.textAlign = hasUrdu ? "right" : "left";
  });

  // Voice recognition needs a language picked before it starts listening
  // (it can't auto-detect like typed text can), so let users tap the
  // language label to choose English or Urdu for their next recording.
  const langDisplay = document.getElementById("currentLanguageDisplay");
  if (langDisplay) {
    langDisplay.style.cursor = "pointer";
    langDisplay.title = "Tap to switch voice input language";
    langDisplay.setAttribute("role", "button");
    langDisplay.tabIndex = 0;
    const toggleLang = () => {
      if (isRecording) return;
      voiceLang = voiceLang === "urdu" ? "english" : "urdu";
      currentLanguage = voiceLang;
      updateLanguageDisplay(currentLanguage);
    };
    langDisplay.addEventListener("click", toggleLang);
    langDisplay.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleLang();
      }
    });
  }
}

function addRipple(e, el) {
  const x = e.clientX - el.getBoundingClientRect().left;
  const y = e.clientY - el.getBoundingClientRect().top;
  const ripple = document.createElement("span");
  ripple.classList.add("ripple-effect");
  ripple.style.left = `${x}px`;
  ripple.style.top = `${y}px`;
  el.appendChild(ripple);
  setTimeout(() => ripple.remove(), 600);
}

// ── Send message ───────────────────────────────────────────────────────--
async function sendMessage(overrideText = null) {
  if (isGenerating) return;

  const input = document.getElementById("userInput");
  const message = overrideText || input.value.trim();
  if (!message) return;

  setGenerating(true);
  input.value = "";

  currentLanguage = detectLanguage(message);
  updateLanguageDisplay(currentLanguage);
  addMessageToChat(message, "user", currentLanguage);
  showTypingIndicator();

  try {
    const data = await askQueryWithWarmupRetry(message, currentLanguage);
    hideTypingIndicator();
    if (data.language) {
      currentLanguage = data.language;
      updateLanguageDisplay(currentLanguage);
    }
    addMessageToChat(data.answer, "system", data.language || currentLanguage, data.sources || []);
  } catch (err) {
    hideTypingIndicator();
    // Warm-up statuses only reach here after every retry has been exhausted.
    const stillWaking = [0, 502, 503, 504].includes(err.status);
    const msg = stillWaking
      ? "The assistant is taking longer than usual to wake up. Please try again in a moment."
      : "Sorry, something went wrong. Please try again.";
    addMessageToChat(msg, "system", "english");
    console.error("Send error:", err);
  } finally {
    setGenerating(false);
  }
}

// On a cold start the backend loads its index/model in the background and
// returns 503 for ~20-30s. Retry a couple of times with backoff before
// surfacing an error, so users don't see a failure for a transient warmup.
// Statuses that mean "the backend isn't ready yet", not "your request was bad":
//   503 - the app is up but still loading its index in the background
//   502 / 504 - the platform proxy has no healthy instance to route to, which
//               is what a spun-down Render free instance returns while waking
//   0   - fetch() failed outright (dropped connection / CORS), see api.js
const WARMUP_STATUSES = [0, 502, 503, 504];

// A free-plan instance sleeps after ~15 min idle and takes ~50s to wake, so the
// backoff has to cover roughly a minute in total rather than a few seconds.
const WARMUP_DELAYS = [3000, 6000, 12000, 20000, 25000];

async function askQueryWithWarmupRetry(message, language) {
  let notified = false;
  for (let attempt = 0; ; attempt++) {
    try {
      return await api.askQuery(message, language);
    } catch (err) {
      const retryable = WARMUP_STATUSES.includes(err.status);
      if (!retryable || attempt >= WARMUP_DELAYS.length) throw err;
      if (!notified) {
        notified = true;
        showWarmupNotice();
      }
      await new Promise((resolve) => setTimeout(resolve, WARMUP_DELAYS[attempt]));
    }
  }
}

// Tell the user why the first message of the day is slow, so a ~50s free-plan
// cold start reads as "waking up" rather than "broken". Shown at most once per
// send, and only once we've actually hit a warm-up failure.
function showWarmupNotice() {
  addMessageToChat(
    "The assistant is waking up — this can take up to a minute on the free tier. Hang tight…",
    "system",
    "english"
  );
}

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

// ── Render a chat message ──────────────────────────────────────────────--
function addMessageToChat(message, sender, language = "english", sources = []) {
  const container = document.getElementById("chatMessages");
  const div = document.createElement("div");
  const isUrdu = language === "urdu";
  const isSystem = sender === "system";

  div.className = `message ${sender}-message${isUrdu ? " urdu-text" : ""}`;
  const displayMsg = isUrdu && isSystem ? cleanUrduText(message) : message;

  let sourcesHtml = "";
  if (isSystem && sources.length > 0) {
    const items = sources
      .slice(0, 3)
      .map(
        (s) =>
          `<span class="source-tag"><i class="fas fa-bookmark"></i> ${escapeHtml(
            s.topic || "Source"
          )}${s.source ? " — " + escapeHtml(s.source) : ""}</span>`
      )
      .join("");
    sourcesHtml = `<div class="sources-container"><div class="sources-label"><i class="fas fa-book-medical"></i> Sources</div><div class="sources-list">${items}</div></div>`;
  }

  const contentDir = isUrdu ? ' dir="rtl"' : "";
  div.innerHTML = `
    <div class="message-avatar"><i class="fas fa-${isSystem ? "robot" : "user"}"></i></div>
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

// ── Voice recording (browser-based speech-to-text) ─────────────────────--
// Uses the Web Speech API directly in the browser instead of uploading
// audio to the backend, so voice input works regardless of whether the
// backend's Whisper transcription is enabled. Chromium-only (Chrome/Edge);
// the mic button is hidden everywhere else.
const SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

function initVoice() {
  const btn = document.getElementById("voiceButton");
  if (!btn) return;
  if (!SpeechRecognitionImpl) {
    btn.style.display = "none";
    return;
  }
  btn.addEventListener("click", toggleRecording);
}

function toggleRecording() {
  if (isGenerating) return;
  if (isRecording) stopRecording();
  else startRecording();
}

function startRecording() {
  const input = document.getElementById("userInput");
  let finalTranscript = "";

  recognition = new SpeechRecognitionImpl();
  recognition.lang = voiceLang === "urdu" ? "ur-PK" : "en-US";
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const transcript = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalTranscript += transcript;
      else interim += transcript;
    }
    input.value = (finalTranscript + interim).trim();
    input.dispatchEvent(new Event("input"));
  };

  recognition.onerror = (e) => {
    console.error("Speech recognition error:", e.error);
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      addMessageToChat(
        "Microphone access was denied. Please allow microphone permissions or type your question.",
        "system",
        "english"
      );
    } else if (e.error !== "no-speech" && e.error !== "aborted") {
      addMessageToChat("Speech recognition failed. Please type your question instead.", "system", "english");
    }
  };

  recognition.onend = () => {
    isRecording = false;
    updateVoiceUI(false);
    input.focus();
  };

  recognition.start();
  isRecording = true;
  updateVoiceUI(true);
}

function stopRecording() {
  if (recognition) recognition.stop();
  isRecording = false;
  updateVoiceUI(false);
}

function updateVoiceUI(recording) {
  const btn = document.getElementById("voiceButton");
  if (!btn) return;
  btn.classList.toggle("recording", recording);
  btn.setAttribute("aria-label", recording ? "Stop recording" : "Start voice recording");
  btn.innerHTML = recording
    ? '<i class="fas fa-stop" aria-hidden="true"></i>'
    : '<i class="fas fa-microphone" aria-hidden="true"></i>';
}

// ── Predefined questions ───────────────────────────────────────────────--
// Built-in suggestions shown whenever the backend can't supply questions
// (empty list, or an error such as the RAG service being unavailable), so the
// panel stays usable instead of dead-ending on a "Could not load" message.
const FALLBACK_QUESTIONS = {
  english: [
    { question: "What are the common symptoms of breast cancer?", category: "symptoms", icon: "fas fa-stethoscope" },
    { question: "How can I manage pain during treatment?", category: "pain", icon: "fas fa-hand-holding-medical" },
    { question: "How do I cope with anxiety and fear?", category: "emotional", icon: "fas fa-heart" },
    { question: "What exercises are safe during recovery?", category: "exercise", icon: "fas fa-dumbbell" },
    { question: "Which foods support recovery and immunity?", category: "nutrition", icon: "fas fa-apple-alt" },
    { question: "What are the treatment options for breast cancer?", category: "general", icon: "fas fa-notes-medical" },
  ],
  urdu: [
    { question: "بریسٹ کینسر کی عام علامات کیا ہیں؟", category: "symptoms", icon: "fas fa-stethoscope" },
    { question: "علاج کے دوران درد کو کیسے کم کریں؟", category: "pain", icon: "fas fa-hand-holding-medical" },
    { question: "پریشانی اور خوف سے کیسے نمٹیں؟", category: "emotional", icon: "fas fa-heart" },
    { question: "صحت یابی کے دوران کون سی ورزشیں محفوظ ہیں؟", category: "exercise", icon: "fas fa-dumbbell" },
    { question: "کون سی غذائیں صحت یابی میں مدد دیتی ہیں؟", category: "nutrition", icon: "fas fa-apple-alt" },
    { question: "بریسٹ کینسر کے علاج کے کیا طریقے ہیں؟", category: "general", icon: "fas fa-notes-medical" },
  ],
};

async function loadPredefinedQuestions(lang) {
  const containerId = lang === "urdu" ? "urdu-questions" : "english-questions";
  const container = document.getElementById(containerId);
  if (!container) return;

  try {
    const data = await api.predefinedQuestions(lang);
    const questions =
      data.questions && data.questions.length ? data.questions : FALLBACK_QUESTIONS[lang];
    renderQuestions(container, questions, lang);
  } catch (e) {
    console.warn(`Failed to load ${lang} questions, using built-in fallback:`, e);
    renderQuestions(container, FALLBACK_QUESTIONS[lang], lang);
  }
}

function renderQuestions(container, questions, lang) {
  container.innerHTML = "";
  questions.forEach((q) => {
    const card = document.createElement("div");
    card.className = `question-card ${lang === "urdu" ? "urdu-text" : ""}`;
    if (lang === "urdu") card.dir = "rtl";
    card.setAttribute("role", "button");
    card.tabIndex = 0;
    card.innerHTML = `
      <div class="card-icon ${q.category || "general"}" aria-hidden="true">
        <i class="${q.icon || "fas fa-question-circle"}"></i>
      </div>
      <div class="card-content"><h3>${escapeHtml(q.question)}</h3></div>`;
    const fire = () => askPredefined(q.question, lang);
    card.addEventListener("click", fire);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fire();
      }
    });
    container.appendChild(card);
  });
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

// ── Tabs ───────────────────────────────────────────────────────────────--
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const content = document.getElementById(`${btn.dataset.tab}-tab`);
      if (content) content.classList.add("active");
    });
  });
}

// ── Language detection (client-side hint) ──────────────────────────────--
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
  if (/[؀-ۿݐ-ݿࢠ-ࣿ]/.test(text)) return "urdu";
  const words = text.toLowerCase().split(/\s+/);
  let count = 0;
  for (const w of words) if (ROMAN_URDU_WORDS.has(w)) count++;
  return count >= 2 ? "urdu" : "english";
}

function updateLanguageDisplay(lang) {
  const el = document.getElementById("currentLanguageDisplay");
  if (el) el.textContent = lang === "urdu" ? "اردو" : "English";
  const stat = document.getElementById("activeLanguage");
  if (stat) stat.textContent = lang === "urdu" ? "اردو" : "English";
}

// ── Urdu text cleaning ─────────────────────────────────────────────────--
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
  for (const [wrong, right] of Object.entries(fixes)) text = text.replaceAll(wrong, right);
  text = text.replace(/[ऀ-ॿ]/g, "");
  text = text.replace(/[一-鿿]/g, "");
  text = text.replace(/[Ḁ-ỿ]/g, "");
  text = text.replace(/[̀-ͯ]/g, "");
  return text.replace(/\s+/g, " ").trim();
}

// ── Helpers ────────────────────────────────────────────────────────────--
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
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}
function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str == null ? "" : String(str);
  return d.innerHTML;
}
