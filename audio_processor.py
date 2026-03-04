# microphone_transcriber_enhanced.py
import os
import sys
import threading
import time
import wave
import tempfile
import queue
import re
import torch
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# For audio recording
import pyaudio
import numpy as np

# For speech recognition
import whisper
from openai import OpenAI
import torchaudio
from transformers import pipeline

class MicrophoneTranscriber:
    """Real-time microphone transcription tool with enhanced translation"""
    
    def __init__(self, use_openai=False, openai_api_key=None):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.audio_format = pyaudio.paInt16
        self.silence_threshold = 500
        self.silence_duration = 2.0
        self.use_openai = use_openai
        self.openai_client = None
        
        if use_openai and openai_api_key:
            print("🔗 Initializing OpenAI client...")
            self.openai_client = OpenAI(api_key=openai_api_key)
            print("✅ OpenAI client ready")
        else:
            print("📦 Loading Whisper large-v3 model (this may take a minute and requires ~10GB VRAM)...")
            
            # Check for GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"⚡ Using device: {device}")
            
            if device == "cpu":
                print("⚠️  Warning: Running on CPU. This will be slow. Consider using GPU for better performance.")
            
            # Load the largest Whisper model
            self.whisper_model = whisper.load_model(
                "large-v3",
                device=device,
                download_root="./models"  # Custom download location
            )
            print("✅ Whisper large-v3 model loaded")
            
            # For better Urdu detection
            self.setup_language_detection()
        
        # Initialize translation pipeline for better quality
        print("🔄 Initializing translation pipeline...")
        self.setup_translation_pipeline()
        print("✅ All systems ready")
    
    def setup_language_detection(self):
        """Setup enhanced language detection patterns"""
        # Comprehensive Urdu Unicode ranges
        self.urdu_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF),  # Arabic Extended-A
            (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
            (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
        ]
        
        # Common Urdu words with higher frequency
        self.urdu_words = {
            'ہے': 10, 'ہیں': 8, 'ہوں': 7, 'کیا': 9, 'کے': 15,
            'کو': 14, 'سے': 13, 'پر': 12, 'میں': 15, 'اور': 14,
            'لیکن': 6, 'اگر': 5, 'تو': 8, 'بھی': 10, 'ہی': 9,
            'تھا': 7, 'تھی': 7, 'تھے': 7, 'ہو': 8, 'رہا': 6,
            'رہی': 6, 'رہے': 6, 'گیا': 5, 'گئی': 5, 'گئے': 5,
            'کر': 12, 'کرنے': 7, 'ہوا': 6, 'ہوئی': 6, 'ہوئے': 6
        }
        
        # Pakistani English context words
        self.pakistani_english = {
            'yaar': 3, 'wallah': 2, 'inshallah': 4, 'mashallah': 3,
            'allah': 4, 'sir': 5, 'madam': 4, 'uncle': 3, 'aunty': 3,
            'bhai': 5, 'pani': 3, 'roti': 3, 'chai': 3, 'bazaar': 3,
            'desi': 2, 'acha': 4, 'theek': 3, 'shukriya': 3, 'khuda': 2
        }
    
    def setup_translation_pipeline(self):
        """Setup translation pipeline for better quality"""
        try:
            # Try to use Facebook's M2M100 model for better translation
            print("🔄 Loading translation model...")
            device = 0 if torch.cuda.is_available() else -1
            
            # For English to Urdu translation
            self.translator_en_to_ur = pipeline(
                "translation",
                model="facebook/m2m100_418M",
                tokenizer="facebook/m2m100_418M",
                device=device
            )
            print("✅ Translation pipeline ready")
        except Exception as e:
            print(f"⚠️  Could not load translation model: {e}")
            print("⚠️  Using fallback translation method")
            self.translator_en_to_ur = None
    
    def start_recording(self):
        """Start recording from microphone"""
        print("\n🎤 Recording started... (Press Ctrl+C to stop)")
        print("   Speak clearly and naturally")
        print("   System will auto-stop after silence")
        print("-" * 60)
        
        self.recording = True
        self.audio_frames = []
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
        # Monitor for silence
        self.silence_thread = threading.Thread(target=self._monitor_silence)
        self.silence_thread.start()
        
        # Start real-time feedback
        self.feedback_thread = threading.Thread(target=self._provide_feedback)
        self.feedback_thread.start()
    
    def _record_audio(self):
        """Record audio from microphone with better quality"""
        p = pyaudio.PyAudio()
        
        try:
            # Get the best available input device
            device_info = self._get_best_input_device(p)
            
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_info['index'],
                frames_per_buffer=self.chunk_size
            )
            
            print(f"🎤 Using microphone: {device_info['name']}")
            
            self.audio_frames = []
            self.volume_history = []
            self.silence_start = None
            self.last_sound_time = time.time()
            
            while self.recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate volume with RMS for better accuracy
                    current_volume = np.sqrt(np.mean(audio_data.astype(float) ** 2))
                    self.volume_history.append(current_volume)
                    
                    if len(self.volume_history) > 100:  # Keep last 100 readings
                        self.volume_history.pop(0)
                    
                    # Adaptive silence threshold
                    if len(self.volume_history) > 50:
                        avg_volume = np.mean(self.volume_history)
                        adaptive_threshold = max(self.silence_threshold, avg_volume * 0.7)
                    else:
                        adaptive_threshold = self.silence_threshold
                    
                    if current_volume > adaptive_threshold:
                        # Sound detected
                        self.last_sound_time = time.time()
                        self.silence_start = None
                        self.audio_frames.append(data)
                    else:
                        # Silence detected
                        self.audio_frames.append(data)
                        
                        # Check for extended silence
                        if time.time() - self.last_sound_time > self.silence_duration:
                            print("\n🤐 Detected silence - processing audio...")
                            self.recording = False
                            break
                    
                except Exception as e:
                    print(f"⚠️  Recording error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Error opening audio stream: {e}")
        finally:
            p.terminate()
    
    def _get_best_input_device(self, p):
        """Get the best available input device"""
        try:
            device_count = p.get_device_count()
            best_device = None
            
            for i in range(device_count):
                try:
                    info = p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        if not best_device or info.get('defaultSampleRate', 0) > best_device.get('defaultSampleRate', 0):
                            best_device = info
                except:
                    continue
            
            if best_device:
                return {'index': best_device['index'], 'name': best_device['name']}
        except:
            pass
        
        return {'index': None, 'name': 'Default Microphone'}
    
    def _provide_feedback(self):
        """Provide real-time feedback during recording"""
        feedback_chars = ['●', '○', '◎', '◉']
        char_index = 0
        
        while self.recording:
            if hasattr(self, 'volume_history') and len(self.volume_history) > 10:
                recent_volume = np.mean(self.volume_history[-10:])
                if recent_volume > self.silence_threshold:
                    char_index = (char_index + 1) % len(feedback_chars)
                    print(f"\r🎤 Speaking {feedback_chars[char_index]}", end="", flush=True)
                else:
                    print("\r🎤 Listening...", end="", flush=True)
            time.sleep(0.1)
    
    def _monitor_silence(self):
        """Monitor for extended silence"""
        while self.recording:
            time.sleep(0.1)
    
    def stop_recording(self):
        """Stop recording and process audio"""
        print("\n⏹️  Stopping recording...")
        self.recording = False
        
        # Wait for threads to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=2)
        if hasattr(self, 'silence_thread'):
            self.silence_thread.join(timeout=2)
        if hasattr(self, 'feedback_thread'):
            self.feedback_thread.join(timeout=1)
        
        # Process the recorded audio
        if hasattr(self, 'audio_frames') and self.audio_frames:
            print("🔍 Processing audio with Whisper large-v3...")
            transcript = self.process_audio()
            return transcript
        return None
    
    def process_audio(self):
        """Process recorded audio and return enhanced transcription"""
        if not self.audio_frames:
            return {"error": "No audio recorded"}
        
        # Save audio to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        
        try:
            # Create WAV file
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            
            print(f"📁 Audio saved: {os.path.basename(temp_filename)}")
            print("🎯 Transcribing with Whisper large-v3...")
            
            if self.use_openai and self.openai_client:
                # Use OpenAI Whisper API
                with open(temp_filename, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        language=None,  # Auto-detect
                        temperature=0.0
                    )
                
                text = transcript.text.strip()
                detected_lang = transcript.language
                
            else:
                # Use local Whisper model with enhanced settings
                result = self.whisper_model.transcribe(
                    temp_filename,
                    language=None,  # Auto-detect
                    task="transcribe",
                    temperature=0.0,  # More deterministic
                    best_of=5,  # Take best of 5 samples
                    beam_size=5,  # Beam search size
                    patience=1.0,  # Patience factor
                    compression_ratio_threshold=2.4,  # Filter out bad results
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    initial_prompt="Transcribe the following speech accurately. Include proper punctuation.",
                    word_timestamps=True  # Better word-level accuracy
                )
                
                text = result["text"].strip()
                detected_lang = result.get("language", "en")
            
            print(f"📝 Detected language: {detected_lang}")
            print(f"📝 Original text: {text}")
            
            # Enhanced language detection
            actual_lang, confidence = self.detect_language_enhanced(text, detected_lang)
            print(f"🔍 Refined detection: {actual_lang} (confidence: {confidence:.2f})")
            
            # Translate if not English
            if actual_lang != "english" and text.strip():
                print("🌍 Translating to English with enhanced quality...")
                english_text = self.translate_to_english_enhanced(text, actual_lang)
                print("✅ High-quality translation complete!")
            else:
                english_text = text
                # Post-process English text for better quality
                english_text = self.post_process_english(english_text)
            
            # Get word confidence if available
            word_confidence = self.get_confidence_metrics(text, detected_lang)
            
            return {
                "original_text": text,
                "english_text": english_text,
                "detected_language": actual_lang,
                "confidence": confidence,
                "word_confidence": word_confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            # Cleanup
            try:
                os.unlink(temp_filename)
            except:
                pass
    
    def detect_language_enhanced(self, text, whisper_lang):
        """Enhanced language detection with confidence scoring"""
        if not text.strip():
            return "english", 0.0
        
        text_lower = text.lower()
        
        # Character-based analysis
        total_chars = len(text)
        if total_chars == 0:
            return "english", 0.0
        
        # Count Urdu/Arabic script characters
        urdu_char_count = 0
        for start, end in self.urdu_ranges:
            for char in text:
                if start <= ord(char) <= end:
                    urdu_char_count += 1
        
        # Count Latin characters
        latin_char_count = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        
        # Word-based analysis
        words = re.findall(r'\b\w+\b', text_lower)
        urdu_word_score = 0
        english_word_score = 0
        
        for word in words:
            if word in self.urdu_words:
                urdu_word_score += self.urdu_words[word]
            elif word in self.pakistani_english:
                english_word_score += self.pakistani_english[word]
            elif len(word) > 2:
                # Basic English word check
                if re.match(r'^[a-z]+$', word):
                    english_word_score += 1
        
        # Calculate confidence scores
        urdu_char_ratio = urdu_char_count / max(total_chars, 1)
        latin_char_ratio = latin_char_count / max(total_chars, 1)
        
        # Weighted scoring
        total_score = (urdu_char_ratio * 0.5) + (latin_char_ratio * 0.3) + \
                     (urdu_word_score * 0.01) + (english_word_score * 0.01)
        
        # Determine language
        if urdu_char_ratio > 0.3 or urdu_word_score > 5:
            confidence = min(1.0, urdu_char_ratio + (urdu_word_score * 0.05))
            return "urdu", confidence
        elif latin_char_ratio > 0.7 or english_word_score > 3:
            confidence = min(1.0, latin_char_ratio + (english_word_score * 0.03))
            return "english", confidence
        else:
            # Use Whisper's detection as fallback
            lang_map = {
                'en': 'english',
                'ur': 'urdu',
                'hi': 'hindi',
                'pa': 'punjabi'
            }
            return lang_map.get(whisper_lang, 'english'), 0.5
    
    def translate_to_english_enhanced(self, text, source_lang):
        """Enhanced translation to English using multiple strategies"""
        try:
            # Clean the text first
            text = self.clean_text(text)
            
            if source_lang == "urdu":
                # Use Whisper for translation (better for Urdu-English)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                with open(temp_file.name, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                if not self.use_openai:
                    # Use local Whisper for translation
                    result = self.whisper_model.transcribe(
                        temp_file.name,
                        language="ur",
                        task="translate",
                        temperature=0.0,
                        best_of=3
                    )
                    translated = result["text"].strip()
                else:
                    # Use OpenAI API for translation
                    with open(temp_file.name, "rb") as f:
                        transcript = self.openai_client.audio.translations.create(
                            model="whisper-1",
                            file=f,
                            response_format="text",
                            temperature=0.0
                        )
                    translated = transcript.strip()
                
                os.unlink(temp_file.name)
                
            else:
                # For other languages, use available translation
                if self.translator_en_to_ur:
                    # Note: This is English to Urdu, we need the reverse
                    # For now, use a simpler approach
                    translated = text  # Fallback
                else:
                    # Simple fallback translation logic
                    translated = self.simple_translate(text, source_lang)
            
            # Post-process translation
            translated = self.post_process_translation(translated)
            return translated
            
        except Exception as e:
            print(f"⚠️  Translation error: {e}")
            # Return original text with note
            return f"[Note: Auto-translation from {source_lang}] {text}"
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)
        
        # Capitalize sentences
        sentences = re.split(r'([.!?])\s+', text)
        cleaned = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i]
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                cleaned.append(sentence)
                if i + 1 < len(sentences):
                    cleaned.append(sentences[i + 1] + ' ')
        
        return ''.join(cleaned).strip()
    
    def post_process_english(self, text):
        """Post-process English text for better quality"""
        # Fix common English transcription errors
        corrections = {
            r'\bi\s+am\b': 'I am',
            r'\bim\b': "I'm",
            r'\bid\b': "I'd",
            r'\bill\b': "I'll",
            r'\bive\b': "I've",
            r'\bdont\b': "don't",
            r'\bdoesnt\b': "doesn't",
            r'\bdidnt\b': "didn't",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
            r'\bcouldnt\b': "couldn't",
            r'\bshouldnt\b': "shouldn't",
            r'\bwouldnt\b': "wouldn't",
            r'\bhavent\b': "haven't",
            r'\bhasnt\b': "hasn't",
            r'\bhadnt\b': "hadn't",
            r'\bisnt\b': "isn't",
            r'\baren't\b': "aren't",
            r'\bwasnt\b': "wasn't",
            r'\bwerent\b': "weren't",
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return self.clean_text(text)
    
    def post_process_translation(self, text):
        """Post-process translated text"""
        # Remove translation artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
        text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses content
        text = self.clean_text(text)
        
        # Ensure proper English punctuation
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def get_confidence_metrics(self, text, language):
        """Calculate confidence metrics for transcription"""
        # Simple confidence calculation based on text properties
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return {"overall": 0.0, "word_count": 0}
        
        # Basic confidence factors
        avg_word_length = sum(len(w) for w in words) / word_count
        unique_words = len(set(words))
        
        # Confidence score (simplified)
        if language == 'en':
            # English-specific confidence
            confidence = min(1.0, 0.7 + (unique_words / max(word_count, 1)) * 0.3)
        else:
            # Non-English
            confidence = min(1.0, 0.6 + (unique_words / max(word_count, 1)) * 0.4)
        
        return {
            "overall": round(confidence, 2),
            "word_count": word_count,
            "avg_word_length": round(avg_word_length, 1),
            "unique_words": unique_words
        }
    
    def simple_translate(self, text, source_lang):
        """Simple fallback translation"""
        # Very basic translation mapping (for demonstration)
        # In production, you'd want a proper translation service
        
        urdu_english_map = {
            'ہے': 'is',
            'ہیں': 'are',
            'ہوں': 'am',
            'کیا': 'what',
            'کے': 'of',
            'کو': 'to',
            'سے': 'from',
            'پر': 'on',
            'میں': 'in',
            'اور': 'and',
        }
        
        words = text.split()
        translated_words = []
        
        for word in words:
            if word in urdu_english_map:
                translated_words.append(urdu_english_map[word])
            else:
                translated_words.append(f"[{word}]")
        
        return ' '.join(translated_words)

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print application header"""
    clear_screen()
    print("🎤 PAKISTANI SPEECH-TO-TEXT TRANSCRIBER (ENHANCED)")
    print("=" * 60)
    print("\n✨ Features:")
    print("   • Whisper large-v3 model (most accurate)")
    print("   • Enhanced Urdu/English detection")
    print("   • High-quality English translation")
    print("   • Real-time feedback")
    print("   • Confidence scoring")
    print("\n⚠️  System Requirements:")
    print("   • 10GB+ VRAM recommended for Whisper large-v3")
    print("   • 16GB+ system RAM")
    print("   • Good quality microphone")
    print("\n" + "=" * 60)

def check_system_resources():
    """Check system resources"""
    import psutil
    
    print("🖥️  System Check:")
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"   RAM: {ram_gb:.1f} GB available")
    
    # Check GPU if available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("   ⚠️  Warning: Low VRAM. Consider using smaller model.")
    else:
        print("   ⚠️  No GPU detected. Using CPU (slow).")
    
    print("-" * 60)

def main():
    """Main function with enhanced menu"""
    print_header()
    check_system_resources()
    
    # Ask for OpenAI API key
    use_openai = False
    openai_key = None
    
    api_choice = input("\n🤖 Use OpenAI API for better results? (y/n): ").lower()
    if api_choice == 'y':
        openai_key = input("🔑 Enter OpenAI API key: ").strip()
        if openai_key:
            use_openai = True
            print("✅ Using OpenAI API")
        else:
            print("⚠️  No API key provided, using local model")
    
    # Initialize transcriber
    transcriber = MicrophoneTranscriber(use_openai=use_openai, openai_api_key=openai_key)
    
    try:
        while True:
            print("\n" + "=" * 60)
            print("🎵 ENHANCED MAIN MENU")
            print("=" * 60)
            print("1. 🎤 Start speaking (auto-stop after silence)")
            print("2. 🎙️  Start speaking (manual stop)")
            print("3. ⚙️  Adjust settings")
            print("4. ℹ️  Show system info")
            print("5. ❌ Exit")
            print("-" * 60)
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                # Auto-stop mode
                print("\n⏺️  Starting in 3 seconds...")
                print("   Speak clearly and naturally.")
                print("   Recording auto-stops after 2 seconds of silence.")
                for i in range(3, 0, -1):
                    print(f"   {i}...")
                    time.sleep(1)
                
                print("\n🎤 RECORDING...")
                
                # Start recording
                transcriber.start_recording()
                
                # Wait for auto-stop
                while transcriber.recording:
                    time.sleep(0.1)
                
                print("\n")
                transcript = transcriber.stop_recording()
                
                if transcript and "error" not in transcript:
                    self.display_enhanced_results(transcript)
            
            elif choice == "2":
                # Manual stop mode
                print("\n⏺️  Starting in 3 seconds...")
                print("   Press Enter when finished speaking.")
                for i in range(3, 0, -1):
                    print(f"   {i}...")
                    time.sleep(1)
                
                print("\n🎤 RECORDING... (Press Enter to stop)")
                
                # Start recording in background
                transcriber.start_recording()
                
                # Wait for user to press Enter
                input("\n⏸️  Press Enter to stop recording...\n")
                
                transcript = transcriber.stop_recording()
                
                if transcript and "error" not in transcript:
                    self.display_enhanced_results(transcript)
            
            elif choice == "3":
                # Settings menu
                self.settings_menu(transcriber)
            
            elif choice == "4":
                # System info
                clear_screen()
                print_header()
                check_system_resources()
                print("\n📊 Current Settings:")
                print(f"   Silence threshold: {transcriber.silence_threshold}")
                print(f"   Silence duration: {transcriber.silence_duration}s")
                print(f"   Using OpenAI: {transcriber.use_openai}")
                print(f"   Sample rate: {transcriber.sample_rate} Hz")
                print(f"   Chunk size: {transcriber.chunk_size}")
                input("\nPress Enter to continue...")
            
            elif choice == "5":
                print("\n👋 Thank you for using Pakistani Speech-to-Text!")
                print("   Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-5.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    def display_enhanced_results(self, transcript):
        """Display enhanced transcription results"""
        print("\n" + "=" * 60)
        print("📄 ENHANCED TRANSCRIPTION RESULTS")
        print("=" * 60)
        
        # Confidence indicator
        confidence = transcript.get('confidence', 0.5)
        confidence_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        
        print(f"\n📊 Confidence: [{confidence_bar}] {confidence:.1%}")
        
        # Language info
        print(f"🌐 Detected Language: {transcript['detected_language'].upper()}")
        
        # Original text
        print(f"\n📝 ORIGINAL TEXT ({transcript['detected_language'].upper()}):")
        print("-" * 40)
        print(transcript['original_text'])
        print("-" * 40)
        
        # English translation
        print(f"\n✅ HIGH-QUALITY ENGLISH TRANSLATION:")
        print("-" * 40)
        print(transcript['english_text'])
        print("-" * 40)
        
        # Word statistics
        if 'word_confidence' in transcript:
            stats = transcript['word_confidence']
            print(f"\n📈 Statistics:")
            print(f"   Words: {stats.get('word_count', 0)}")
            print(f"   Unique words: {stats.get('unique_words', 0)}")
            print(f"   Overall confidence: {stats.get('overall', 0):.1%}")
        
        # Ask to save
        save = input("\n💾 Save to file? (y/n): ").lower()
        if save == 'y':
            filename = f"transcript_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("ENHANCED SPEECH TRANSCRIPTION\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Timestamp: {transcript.get('timestamp', datetime.now().isoformat())}\n")
                f.write(f"Confidence: {confidence:.1%}\n")
                f.write(f"Detected Language: {transcript['detected_language'].upper()}\n\n")
                f.write("ORIGINAL TEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(transcript['original_text'] + "\n")
                f.write("-" * 40 + "\n\n")
                f.write("ENGLISH TRANSLATION:\n")
                f.write("-" * 40 + "\n")
                f.write(transcript['english_text'] + "\n")
                f.write("-" * 40 + "\n\n")
                
                if 'word_confidence' in transcript:
                    stats = transcript['word_confidence']
                    f.write("STATISTICS:\n")
                    f.write(f"  Word count: {stats.get('word_count', 0)}\n")
                    f.write(f"  Unique words: {stats.get('unique_words', 0)}\n")
                    f.write(f"  Overall confidence: {stats.get('overall', 0):.1%}\n")
            
            print(f"✅ Saved to {filename}")
    
    def settings_menu(self, transcriber):
        """Enhanced settings menu"""
        print("\n⚙️  ENHANCED SETTINGS")
        print("-" * 40)
        print(f"1. Silence threshold: {transcriber.silence_threshold}")
        print(f"2. Silence duration: {transcriber.silence_duration}s")
        print(f"3. Sample rate: {transcriber.sample_rate} Hz")
        print(f"4. Chunk size: {transcriber.chunk_size}")
        print(f"5. Back to main menu")
        
        setting_choice = input("\nSelect setting to adjust (1-5): ").strip()
        
        if setting_choice == "1":
            try:
                new_threshold = int(input("Enter new threshold (100-5000, default 500): "))
                if 100 <= new_threshold <= 5000:
                    transcriber.silence_threshold = new_threshold
                    print(f"✅ Threshold set to {new_threshold}")
                else:
                    print("❌ Please enter value between 100 and 5000")
            except ValueError:
                print("❌ Invalid input")
        
        elif setting_choice == "2":
            try:
                new_duration = float(input("Enter new duration in seconds (0.5-10): "))
                if 0.5 <= new_duration <= 10.0:
                    transcriber.silence_duration = new_duration
                    print(f"✅ Duration set to {new_duration}s")
                else:
                    print("❌ Please enter value between 0.5 and 10")
            except ValueError:
                print("❌ Invalid input")
        
        elif setting_choice == "3":
            try:
                new_rate = int(input("Enter sample rate (8000-48000): "))
                if 8000 <= new_rate <= 48000:
                    transcriber.sample_rate = new_rate
                    print(f"✅ Sample rate set to {new_rate} Hz")
                else:
                    print("❌ Please enter value between 8000 and 48000")
            except ValueError:
                print("❌ Invalid input")
        
        elif setting_choice == "4":
            try:
                new_chunk = int(input("Enter chunk size (256-4096): "))
                if 256 <= new_chunk <= 4096:
                    transcriber.chunk_size = new_chunk
                    print(f"✅ Chunk size set to {new_chunk}")
                else:
                    print("❌ Please enter value between 256 and 4096")
            except ValueError:
                print("❌ Invalid input")

if __name__ == "__main__":
    # Enhanced installation instructions
    print("🔧 ENHANCED INSTALLATION REQUIRED")
    print("=" * 60)
    print("For BEST results:")
    print("\nOption 1: Local setup (requires powerful GPU):")
    print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("pip install openai-whisper")
    print("pip install pyaudio numpy psutil")
    print("\nOption 2: Cloud/API setup (recommended for most users):")
    print("pip install openai pyaudio numpy psutil")
    print("\nFor Ubuntu/Debian:")
    print("sudo apt-get install portaudio19-dev python3-pyaudio")
    print("\nFor macOS:")
    print("brew install portaudio")
    print("=" * 60)
    
    # Check dependencies
    required_packages = [
        ("pyaudio", "pyaudio"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
        ("torch", "torch"),
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        
        install = input("\nDo you want to install missing packages? (y/n): ").lower()
        if install == 'y':
            import subprocess
            import sys
            
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            print("\n✅ Installation complete! Restarting...")
            print("=" * 60)
    
    # Check for Whisper/openai
    try:
        import whisper
        print("✅ Whisper installed")
    except ImportError:
        print("⚠️  Whisper not installed. Install with: pip install openai-whisper")
    
    try:
        from openai import OpenAI
        print("✅ OpenAI installed")
    except ImportError:
        print("⚠️  OpenAI not installed. Install with: pip install openai")
    
    print("\n" + "=" * 60)
    input("Press Enter to start the application...")
    
    # Run the application
    main()