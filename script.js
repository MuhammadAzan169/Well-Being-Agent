// script.js - Voice queries now return text responses only
// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const voiceButton = document.getElementById('voiceButton');
const typingIndicator = document.getElementById('typingIndicator');
const messageCount = document.getElementById('messageCount');
const activeLanguage = document.getElementById('activeLanguage');
const currentLanguageDisplay = document.getElementById('currentLanguageDisplay');
const welcomeTime = document.getElementById('welcomeTime');
const englishQuestions = document.getElementById('english-questions');
const urduQuestions = document.getElementById('urdu-questions');

// State Management
let currentLanguage = 'english';
let messageCounter = 0;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    setWelcomeTime();
    initializeWelcomeMessage();
});

async function initializeApp() {
    updateLanguageDisplay();
    updateInputPlaceholder();
    await loadPredefinedQuestions();
    updateQuestionsDisplay();
}

function setupEventListeners() {
    // Send message on button click
    sendButton.addEventListener('click', sendMessage);
    
    // Send message on Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            switchTab(tabId);
        });
    });

    // Voice button listener
    if (voiceButton) {
        voiceButton.addEventListener('click', async () => {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        });
    }

    userInput.focus();
}

function initializeWelcomeMessage() {
    // Ensure the welcome message has proper styling and is visible
    const welcomeMessage = document.querySelector('.system-message');
    if (welcomeMessage) {
        welcomeMessage.style.opacity = '1';
        welcomeMessage.style.transform = 'translateY(0) scale(1)';
    }
}

function setWelcomeTime() {
    const now = new Date();
    welcomeTime.textContent = now.toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`${tabId}-tab`).classList.add('active');

    const newLanguage = tabId === 'urdu' ? 'urdu' : 'english';
    if (currentLanguage !== newLanguage) {
        currentLanguage = newLanguage;
        updateLanguageDisplay();
        updateInputPlaceholder();
    }
}

function updateLanguageDisplay() {
    activeLanguage.textContent = currentLanguage === 'urdu' ? 'اردو' : 'English';
    currentLanguageDisplay.textContent = currentLanguage === 'urdu' ? 'Urdu' : 'English';
}

function updateInputPlaceholder() {
    if (currentLanguage === 'urdu') {
        userInput.placeholder = 'بریسٹ کینسر کے بارے میں پوچھیں...';
        userInput.style.direction = 'rtl';
        userInput.style.textAlign = 'right';
    } else {
        userInput.placeholder = 'Ask about breast cancer support, treatment options, or recovery...';
        userInput.style.direction = 'ltr';
        userInput.style.textAlign = 'left';
    }
}

async function loadPredefinedQuestions() {
    try {
        const englishResponse = await fetch('/predefined-questions?language=english');
        const englishData = await englishResponse.json();
        if (englishData.status === 'success') {
            window.predefinedEnglishQuestions = englishData.questions;
        }

        const urduResponse = await fetch('/predefined-questions?language=urdu');
        const urduData = await urduResponse.json();
        if (urduData.status === 'success') {
            window.predefinedUrduQuestions = urduData.questions;
        }
    } catch (error) {
        console.error('Error loading predefined questions:', error);
    }
}

function updateQuestionsDisplay() {
    updateQuestionList(englishQuestions, 'english');
    updateQuestionList(urduQuestions, 'urdu');
}

function updateQuestionList(container, language) {
    container.innerHTML = '';
    const predefinedQuestions = language === 'urdu' ? 
        window.predefinedUrduQuestions : 
        window.predefinedEnglishQuestions;
    
    if (predefinedQuestions && predefinedQuestions.length > 0) {
        predefinedQuestions.forEach((questionData) => {
            const questionCard = createQuestionCard(questionData, language);
            container.appendChild(questionCard);
        });
    } else {
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';
        emptyState.innerHTML = `
            <i class="fas fa-comments"></i>
            <p>${language === 'urdu' ? 'ابھی تک کوئی اردو سوالات نہیں ہیں۔' : 'No questions available yet.'}</p>
        `;
        container.appendChild(emptyState);
    }
}

function createQuestionCard(questionData, language) {
    const questionCard = document.createElement('button');
    questionCard.className = `question-card ${language === 'urdu' ? 'urdu-text' : ''} predefined-card`;
    questionCard.setAttribute('data-question', questionData.question);
    const icon = questionData.icon || 'fas fa-question-circle';
    questionCard.innerHTML = `
        <div class="card-icon ${questionData.category || 'general'}">
            <i class="${icon}"></i>
        </div>
        <div class="card-content">
            <h3>${questionData.question}</h3>
        </div>
        <div class="card-arrow">
            <i class="fas fa-chevron-right"></i>
        </div>
    `;
    questionCard.addEventListener('click', function() {
        userInput.value = questionData.question;
        sendMessage();
    });
    return questionCard;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    
    addMessageToChat(message, 'user', currentLanguage);
    userInput.value = '';
    userInput.disabled = true;
    sendButton.disabled = true;
    showTypingIndicator();

    try {
        const response = await fetch('/ask-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: message, language: currentLanguage, response_type: 'text' })
        });
        const data = await response.json();
        hideTypingIndicator();

        if (data.status === 'success') {
            addMessageToChat(data.answer, 'system', data.language);
            updateMessageCount();
        } else {
            addMessageToChat("I'm having trouble processing your request. Please try again.", 'system', 'english');
        }
    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        addMessageToChat("Connection issue. Please try again.", 'system', 'english');
    } finally {
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }
}

function cleanUrduText(text) {
    const urduFixes = {
        // Character fixes
        'ہےہ': 'ہے',
        'مہےں': 'میں', 
        'ہےں': 'ہیں',
        'ھے': 'ہے',
        'ھوں': 'ہوں',
        'ھیں': 'ہیں',
        'ےے': 'ے',
        'ںں': 'ں',
        'ہہ': 'ہ',
        'یی': 'ی',
        
        // Word fixes
        'مجہے': 'مجھے',
        'پروگرہوں': 'پروگرام',
        'کہےنسر': 'کینسر',
        'ڈڈاکٹر': 'ڈاکٹر',
        'کا ے لہےے': 'کے لیے',
        'جسے سے': 'جس سے',
        'اکٹر': 'ڈاکٹر',
        'اکیل': 'اکیلے',
        'میش': 'میں',
        'وتی': 'ہوتی',
        'لکی': 'ہلکی',
        'بتر': 'بہتر',
        
        // Grammar fixes
        'ک دوران': 'کے دوران',
        'ک بار': 'کے بارے',
        'ک بعد': 'کے بعد', 
        'ک لی': 'کے لیے',
        'ک ساتھ': 'کے ساتھ',
        'ک طور': 'کے طور',
        'ک ذریع': 'کے ذریعے',
        'ک مطابق': 'کے مطابق'
    };

    let cleanedText = text;
    
    // Apply all fixes
    Object.keys(urduFixes).forEach(wrong => {
        const regex = new RegExp(escapeRegExp(wrong), 'g');
        cleanedText = cleanedText.replace(regex, urduFixes[wrong]);
    });

    // Fix spacing issues
    cleanedText = cleanedText.replace(/\s+/g, ' ');
    cleanedText = cleanedText.replace(/ \./g, '.');
    cleanedText = cleanedText.replace(/ ،/g, '،');
    cleanedText = cleanedText.replace(/  /g, ' ');
    cleanedText = cleanedText.replace(/۔۔/g, '۔');
    
    return cleanedText.trim();
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function addMessageToChat(message, sender, language = 'english') {
    // Clean Urdu text before displaying
    if (language === 'urdu') {
        message = cleanUrduText(message);
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    if (language === 'urdu') {
        messageDiv.classList.add('urdu-text');
    }
    
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const avatarIcon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="${avatarIcon}"></i>
        </div>
        <div class="message-content ${language === 'urdu' ? 'urdu-text' : ''}">
            <p>${formatMessage(message)}</p>
            <span class="message-time">${timestamp}</span>
            ${language === 'urdu' ? '<div class="language-badge">اردو</div>' : ''}
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // Apply animation to new messages only
    setTimeout(() => {
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0) scale(1)';
    }, 10);
}

function formatMessage(message) {
    return message.replace(/\n/g, '<br>');
}

function showTypingIndicator() {
    typingIndicator.style.display = 'flex';
    scrollToBottom();
}

function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
}

function updateMessageCount() {
    messageCounter++;
    messageCount.textContent = messageCounter;
}

function scrollToBottom() {
    setTimeout(() => {
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    }, 100);
}

// Voice Recording Feature - Now returns text responses only
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('file', audioBlob, 'voiceNote.webm');
            formData.append('language', currentLanguage); // Pass current tab language

            // Add user voice message to chat
            addUserVoiceMessageToChat(audioBlob);
            showTypingIndicator();

            try {
                const response = await fetch('/voice-query', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                hideTypingIndicator();

                if (data.status === 'success') {
                    // ✅ ALWAYS show text response for voice queries
                    if (data.text && data.text.trim() !== '') {
                        addMessageToChat(data.text, 'system', data.language);
                    } else {
                        // Fallback message
                        const fallbackMessage = data.language === 'urdu' 
                            ? "میں آپ کی آواز کا پیغام سمجھ گئی ہوں۔ آپ کیسے مدد کر سکتی ہوں؟"
                            : "I've processed your voice message. How can I help you further?";
                        addMessageToChat(fallbackMessage, 'system', data.language);
                    }
                    
                    updateMessageCount();
                } else {
                    const errorMessage = currentLanguage === 'urdu'
                        ? "معذرت، آپ کی آواز کا پیغام پروسیس نہیں کر سکی۔"
                        : "Sorry, couldn't process your voice message.";
                    addMessageToChat(errorMessage, 'system', currentLanguage);
                }
            } catch (err) {
                console.error('Voice query error:', err);
                hideTypingIndicator();
                const errorMessage = currentLanguage === 'urdu'
                    ? "آواز کے پروسیس میں خرابی۔"
                    : "Error processing voice input.";
                addMessageToChat(errorMessage, 'system', currentLanguage);
            }
        };

        mediaRecorder.start();
        isRecording = true;
        voiceButton.classList.add('recording');
        voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
    } catch (err) {
        console.error('Microphone access error:', err);
        const errorMessage = currentLanguage === 'urdu'
            ? 'براہ کرم آواز ریکارڈ کرنے کے لیے مائیکروفون کی رسائی کی اجازت دیں۔'
            : 'Please allow microphone access to record voice messages.';
        alert(errorMessage);
    }
}

function addUserVoiceMessageToChat(audioBlob) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message user-message audio-message`;
    
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const audioUrl = URL.createObjectURL(audioBlob);
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-user"></i>
        </div>
        <div class="message-content">
            <div class="audio-message-container user-audio">
                <div class="audio-player-wrapper">
                    <audio controls class="voice-note-player">
                        <source src="${audioUrl}" type="audio/webm">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                <div class="audio-duration">Your voice message</div>
            </div>
            <span class="message-time">${timestamp}</span>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    setTimeout(() => {
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0) scale(1)';
    }, 10);
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;
    voiceButton.classList.remove('recording');
    voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
}

// Logo animation
const logo = document.querySelector('.logo');
if (logo) {
    setInterval(() => {
        logo.style.transform = 'rotate(5deg)';
        setTimeout(() => { logo.style.transform = 'rotate(-5deg)'; }, 1000);
        setTimeout(() => { logo.style.transform = 'rotate(0deg)'; }, 2000);
    }, 5000);
}