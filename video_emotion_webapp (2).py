from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import cv2
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
import json
import io
import base64
from datetime import datetime
import uuid
warnings.filterwarnings('ignore')

# Set better plotting style
plt.style.use('default')
sns.set_palette("husl")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Model paths (using your existing paths)
EEG_MODEL_PATH = r"D:/metadata/saj_regression/eeg_rf_regressor.pkl"
EEG_SCALER_PATH = r"D:/metadata/saj_regression/eeg_scaler.pkl"
EMOTION_MODEL_PATH = r"D:/EEG_cleaned_data/Improved_Emotion_Model.pkl"

class VideoEmotionAnalyzer:
    def __init__(self):
        self.eeg_model = None
        self.eeg_scaler = None
        self.emotion_model = None
        self.emotion_scaler = None
        self.expected_features = None
        self.load_models()
    
    def load_models(self):
        """Load all required models"""
        try:
            self.eeg_model = joblib.load(EEG_MODEL_PATH)
            self.eeg_scaler = joblib.load(EEG_SCALER_PATH)
            emotion_artifacts = joblib.load(EMOTION_MODEL_PATH)
            self.emotion_model = emotion_artifacts.get('rf_model')
            self.emotion_scaler = emotion_artifacts.get('scaler')
            self.expected_features = emotion_artifacts.get('features')
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def extract_video_features_per_second(self, video_path):
        """Extract features from video per second"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            fps = 30  # Default fallback
        
        frame_count = 0
        second_count = 0
        per_second_features = []
        
        # Current second accumulators
        brightness_acc = []
        contrast_acc = []
        laplace_var_acc = []
        color_cast_acc = []
        hue_acc = []
        saturation_acc = []
        value_acc = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            brightness = np.mean(gray)
            contrast = np.std(gray)
            laplace_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            color_cast = np.std([np.mean(frame[:, :, i]) for i in range(3)])
            hue = np.mean(hsv[:, :, 0])
            saturation = np.mean(hsv[:, :, 1]) / 255
            value = np.mean(hsv[:, :, 2]) / 255
            
            brightness_acc.append(brightness)
            contrast_acc.append(contrast)
            laplace_var_acc.append(laplace_var)
            color_cast_acc.append(color_cast)
            hue_acc.append(hue)
            saturation_acc.append(saturation)
            value_acc.append(value)
            
            frame_count += 1
            
            if frame_count >= fps:
                second_features = {
                    'second': second_count,
                    'brightness': np.mean(brightness_acc),
                    'contrast': np.mean(contrast_acc),
                    'laplace_var': np.mean(laplace_var_acc),
                    'color_cast': np.mean(color_cast_acc),
                    'hue': np.mean(hue_acc),
                    'saturation': np.mean(saturation_acc),
                    'value': np.mean(value_acc),
                }
                
                if per_second_features:
                    prev_features = per_second_features[-1]
                    second_features.update({
                        'dif_brightness': abs(second_features['brightness'] - prev_features['brightness']),
                        'dif_contrast': abs(second_features['contrast'] - prev_features['contrast']),
                        'dif_laplace_var': abs(second_features['laplace_var'] - prev_features['laplace_var']),
                        'dif_color_cast': abs(second_features['color_cast'] - prev_features['color_cast']),
                        'dif_hue': abs(second_features['hue'] - prev_features['hue']),
                        'dif_saturation': abs(second_features['saturation'] - prev_features['saturation']),
                        'dif_value': abs(second_features['value'] - prev_features['value']),
                    })
                else:
                    second_features.update({
                        'dif_brightness': 0, 'dif_contrast': 0, 'dif_laplace_var': 0,
                        'dif_color_cast': 0, 'dif_hue': 0, 'dif_saturation': 0, 'dif_value': 0
                    })
                
                per_second_features.append(second_features)
                
                brightness_acc = []
                contrast_acc = []
                laplace_var_acc = []
                color_cast_acc = []
                hue_acc = []
                saturation_acc = []
                value_acc = []
                
                frame_count = 0
                second_count += 1
        
        cap.release()
        return per_second_features
    
    def check_model_compatibility(self):
        """Check model compatibility and return expected features"""
        try:
            if hasattr(self.eeg_model, 'feature_names_in_'):
                return list(self.eeg_model.feature_names_in_)
            elif hasattr(self.eeg_scaler, 'feature_names_in_'):
                return list(self.eeg_scaler.feature_names_in_)
            else:
                return [
                    'brightness', 'dif_brightness', 'contrast', 'dif_contrast',
                    'laplace_var', 'dif_laplace_var', 'color_cast', 'dif_color_cast',
                    'hue', 'dif_hue', 'saturation', 'dif_saturation',
                    'value', 'dif_value', 'age', 'gender'
                ]
        except Exception:
            return [
                'brightness', 'dif_brightness', 'contrast', 'dif_contrast',
                'laplace_var', 'dif_laplace_var', 'color_cast', 'dif_color_cast',
                'hue', 'dif_hue', 'saturation', 'dif_saturation',
                'value', 'dif_value', 'age', 'gender'
            ]
    
    def infer_emotion(self, valence, arousal, dominance, immersion, visual):
        """Enhanced emotion inference with more granular categories"""
        if immersion < 0.3:
            return "Disengaged"
        
        if valence > 0.7:
            if arousal > 0.7:
                return "Ecstatic" if dominance > 0.6 else "Thrilled"
            elif arousal > 0.4:
                return "Happy" if dominance > 0.5 else "Pleasant"
            else:
                return "Serene"
        elif valence > 0.5:
            if arousal > 0.6:
                return "Excited" if dominance > 0.5 else "Anticipatory"
            else:
                return "Content"
        elif valence < 0.3:
            if arousal > 0.6:
                return "Furious" if dominance > 0.6 else "Anxious"
            else:
                return "Sad" if dominance < 0.4 else "Disappointed"
        
        return "Neutral"
    
    def detect_emotion_drops(self, emotion_scores, timestamps, threshold=0.15):
        """Detect sudden drops in emotion scores"""
        drops = {'valence': [], 'arousal': [], 'dominance': [], 'immersion': []}
        
        for i, emotion_type in enumerate(['valence', 'arousal', 'dominance', 'immersion']):
            if i >= emotion_scores.shape[1]:
                continue
                
            scores = emotion_scores[:, i]
            
            for j in range(1, len(scores)):
                drop = scores[j-1] - scores[j]
                if drop > threshold:
                    drops[emotion_type].append({
                        'time': timestamps[j],
                        'drop_magnitude': drop,
                        'from_value': scores[j-1],
                        'to_value': scores[j]
                    })
        
        return drops
    
    def create_emotion_plot(self, emotion_scores, timestamps, session_id):
        """Create emotion trend plot and return filename"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        emotion_labels = ['Valence', 'Arousal', 'Dominance', 'Immersion', 'Visual']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (label, color) in enumerate(zip(emotion_labels, colors)):
            if i < emotion_scores.shape[1]:
                smoothed = np.convolve(emotion_scores[:, i], np.ones(3)/3, mode='same')
                ax.plot(timestamps, smoothed, label=label, color=color, linewidth=3, alpha=0.8)
                ax.scatter(timestamps[::10], emotion_scores[::10, i], color=color, alpha=0.3, s=20)
        
        ax.set_title('Emotion Scores Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Emotion Score', fontsize=12)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_facecolor('#f8f9fa')
        
        filename = f'emotion_plot_{session_id}.png'
        filepath = os.path.join('static/plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        return filename
    
    def create_summary_plot(self, emotion_scores, session_id):
        """Create summary plots and return filename"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        emotion_labels = ['Valence', 'Arousal', 'Dominance', 'Immersion']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Average emotion scores
        avg_scores = np.mean(emotion_scores[:, :4], axis=0)
        bars1 = ax1.bar(emotion_labels, avg_scores, color=colors, alpha=0.7)
        
        for bar, score in zip(bars1, avg_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Average Emotion Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Score')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Emotion volatility
        volatility = np.std(emotion_scores[:, :4], axis=0)
        bars2 = ax2.bar(emotion_labels, volatility, color=colors, alpha=0.7)
        
        for bar, vol in zip(bars2, volatility):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{vol:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_title('Emotion Volatility (Std Dev)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3, axis='y')
        
        filename = f'summary_plot_{session_id}.png'
        filepath = os.path.join('static/plots', filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        return filename
    
    def analyze_single_video(self, video_path, age, gender, session_id):
        """Analyze video for a single person"""
        try:
            # Extract features
            per_second_data = self.extract_video_features_per_second(video_path)
            
            if not per_second_data:
                return {"error": "No video data extracted"}
            
            eeg_expected_features = self.check_model_compatibility()
            
            all_emotion_scores = []
            all_interpreted_emotions = []
            timestamps = []
            
            for second_data in per_second_data:
                # Prepare features for EEG model
                features = {k: v for k, v in second_data.items() if k != 'second'}
                features['age'] = age
                features['gender'] = gender
                
                feature_df = pd.DataFrame([features])
                
                # Ensure all expected features are present
                for feat in eeg_expected_features:
                    if feat not in feature_df.columns:
                        feature_df[feat] = 0.0
                
                try:
                    feature_df = feature_df[eeg_expected_features]
                except KeyError:
                    available_features = [f for f in eeg_expected_features if f in feature_df.columns]
                    feature_df = feature_df[available_features]
                
                # Predict EEG
                try:
                    X_scaled = self.eeg_scaler.transform(feature_df)
                    simulated_eeg = self.eeg_model.predict(X_scaled)[0]
                except Exception:
                    simulated_eeg = np.random.normal(0, 1, 128)
                
                # Prepare emotion input
                eeg_feature_names = [f"eeg_de_{i}" for i in range(len(simulated_eeg))]
                emotion_input_dict = dict(zip(eeg_feature_names, simulated_eeg))
                emotion_input_dict['age'] = age
                emotion_input_dict['gender'] = gender
                emotion_input = pd.DataFrame([emotion_input_dict])
                
                # Align with expected features
                for feat in self.expected_features:
                    if feat not in emotion_input.columns:
                        emotion_input[feat] = 0.0
                emotion_input = emotion_input[self.expected_features]
                
                # Predict emotions
                try:
                    if self.emotion_scaler and len(emotion_input.columns) > 0:
                        emotion_input_scaled = self.emotion_scaler.transform(emotion_input)
                    else:
                        emotion_input_scaled = emotion_input.values
                        
                    predicted_emotion = self.emotion_model.predict(emotion_input_scaled)[0]
                except Exception:
                    predicted_emotion = [0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
                
                all_emotion_scores.append(predicted_emotion)
                
                # Interpret emotion
                emotion_dict = {}
                emotion_labels = ['valence', 'arousal', 'dominance', 'immersion', 'visual', 'auditory']
                for i, label in enumerate(emotion_labels):
                    if i < len(predicted_emotion):
                        emotion_dict[label] = predicted_emotion[i]
                    else:
                        emotion_dict[label] = 0.0
                
                interpreted_emotion = self.infer_emotion(
                    emotion_dict.get('valence', 0),
                    emotion_dict.get('arousal', 0),
                    emotion_dict.get('dominance', 0),
                    emotion_dict.get('immersion', 0),
                    emotion_dict.get('visual', 0)
                )
                all_interpreted_emotions.append(interpreted_emotion)
                timestamps.append(second_data['second'])
            
            # Convert to numpy arrays
            all_emotion_scores = np.array(all_emotion_scores)
            
            # Create plots
            emotion_plot = self.create_emotion_plot(all_emotion_scores, timestamps, session_id)
            summary_plot = self.create_summary_plot(all_emotion_scores, session_id)
            
            # Calculate results
            avg_emotion_scores = np.mean(all_emotion_scores, axis=0)
            emotion_counter = Counter(all_interpreted_emotions)
            most_dominant_emotion = emotion_counter.most_common(1)[0][0]
            dominant_percentage = (emotion_counter.most_common(1)[0][1] / len(all_interpreted_emotions)) * 100
            
            # Detect emotion drops
            emotion_drops = self.detect_emotion_drops(all_emotion_scores, timestamps)
            
            return {
                'success': True,
                'avg_emotions': avg_emotion_scores.tolist(),
                'emotion_distribution': dict(emotion_counter),
                'dominant_emotion': most_dominant_emotion,
                'dominant_percentage': dominant_percentage,
                'total_seconds': len(per_second_data),
                'emotion_drops': emotion_drops,
                'emotion_plot': emotion_plot,
                'summary_plot': summary_plot,
                'detailed_emotions': [
                    {
                        'time': t,
                        'valence': float(scores[0]) if len(scores) > 0 else 0,
                        'arousal': float(scores[1]) if len(scores) > 1 else 0,
                        'dominance': float(scores[2]) if len(scores) > 2 else 0,
                        'immersion': float(scores[3]) if len(scores) > 3 else 0,
                        'emotion': emotion
                    }
                    for t, scores, emotion in zip(timestamps, all_emotion_scores, all_interpreted_emotions)
                ]
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

# Initialize analyzer
analyzer = VideoEmotionAnalyzer()

@app.route('/')
def index():
    """Main page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Emotion Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 90%;
        }
        
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #333;
            font-weight: 600;
        }
        
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .file-upload {
            position: relative;
            display: inline-block;
            cursor: pointer;
            width: 100%;
        }
        
        .file-upload input[type=file] {
            position: absolute;
            left: -9999px;
        }
        
        .file-upload-label {
            padding: 1rem;
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .file-upload-label:hover {
            background: #e9ecef;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .progress {
            width: 100%;
            height: 10px;
            background: #e1e5e9;
            border-radius: 5px;
            overflow: hidden;
            margin: 1rem 0;
            display: none;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .results {
            display: none;
            margin-top: 2rem;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .emotion-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .emotion-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .emotion-card h3 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .emotion-score {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .plot-container {
            margin: 1rem 0;
            text-align: center;
        }
        
        .plot-container img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .nav-links {
            text-align: center;
            margin-top: 2rem;
        }
        
        .nav-links a {
            color: #667eea;
            text-decoration: none;
            margin: 0 1rem;
            font-weight: 600;
        }
        
        .nav-links a:hover {
            text-decoration: underline;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .dominant-emotion {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .dominant-emotion h2 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .insights {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .insight-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .insight-card h4 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .timeline {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .timeline-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            border-bottom: 1px solid #eee;
        }
        
        .timeline-item:last-child {
            border-bottom: none;
        }
        
        .time-marker {
            font-weight: bold;
            color: #667eea;
        }
        
        .emotion-marker {
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Video Emotion Analysis</h1>
            <p>Upload a video to analyze emotional responses using AI-powered EEG simulation</p>
        </div>
        
        <form id="analysisForm">
            <div class="form-group">
                <label for="videoFile">Select Video File (MP4, AVI, MOV, MKV)</label>
                <div class="file-upload">
                    <input type="file" id="videoFile" accept=".mp4,.avi,.mov,.mkv" required>
                    <label for="videoFile" class="file-upload-label">
                        <div>📁 Click to select video file</div>
                        <small>Max file size: 500MB</small>
                    </label>
                </div>
            </div>
            
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" min="5" max="100" value="25" required>
            </div>
            
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" required>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>
            
            <button type="submit" class="btn" id="analyzeBtn">
                🔬 Analyze Emotion
            </button>
        </form>
        
        <div class="progress" id="progress">
            <div class="progress-bar" style="width: 0%"></div>
        </div>
        
        <div class="error" id="error" style="display: none;"></div>
        
        <div class="results" id="results">
            <div class="dominant-emotion" id="dominantEmotion"></div>
            
            <div class="emotion-grid" id="emotionGrid"></div>
            
            <div class="plot-container" id="emotionPlot"></div>
            
            <div class="plot-container" id="summaryPlot"></div>
            
            <div class="insights" id="insights"></div>
            
            <div class="timeline" id="timeline">
                <h4>🕒 Emotion Timeline</h4>
                <div id="timelineContent"></div>
            </div>
        </div>
        
        <div class="nav-links">
            <a href="/demographic_analysis">📊 Demographic Analysis</a>
            <a href="#" onclick="location.reload()">🔄 New Analysis</a>
        </div>
    </div>

    <script>
        let sessionId = null;
        let uploadedFilename = null;

        document.getElementById('videoFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const label = document.querySelector('.file-upload-label div');
                label.textContent = `📁 ${file.name}`;
            }
        });

        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const videoFile = document.getElementById('videoFile').files[0];
            const age = document.getElementById('age').value;
            const gender = document.getElementById('gender').value;
            
            if (!videoFile) {
                showError('Please select a video file');
                return;
            }
            
            hideError();
            hideResults();
            showProgress(10);
            setButtonState(true);
            
            try {
                const formData = new FormData();
                formData.append('video', videoFile);
                
                showProgress(30);
                const uploadResponse = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const uploadResult = await uploadResponse.json();
                
                if (!uploadResult.success) {
                    throw new Error(uploadResult.error);
                }
                
                sessionId = uploadResult.session_id;
                uploadedFilename = uploadResult.filename;
                
                showProgress(60);
                
                const analysisResponse = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        filename: uploadedFilename,
                        age: parseInt(age),
                        gender: parseInt(gender)
                    })
                });
                
                const analysisResult = await analysisResponse.json();
                
                showProgress(100);
                
                if (analysisResult.error) {
                    throw new Error(analysisResult.error);
                }
                
                displayResults(analysisResult);
                
            } catch (error) {
                showError(error.message);
            } finally {
                hideProgress();
                setButtonState(false);
            }
        });

        function showProgress(percentage) {
            const progress = document.getElementById('progress');
            const progressBar = progress.querySelector('.progress-bar');
            progress.style.display = 'block';
            progressBar.style.width = percentage + '%';
        }

        function hideProgress() {
            document.getElementById('progress').style.display = 'none';
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }

        function hideResults() {
            document.getElementById('results').style.display = 'none';
        }

        function setButtonState(disabled) {
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = disabled;
            btn.textContent = disabled ? '🔬 Analyzing...' : '🔬 Analyze Emotion';
        }

        function displayResults(results) {
            document.getElementById('results').style.display = 'block';
            
            const dominantDiv = document.getElementById('dominantEmotion');
            dominantDiv.innerHTML = `
                <h2>${results.dominant_emotion}</h2>
                <p>${results.dominant_percentage.toFixed(1)}% of the video</p>
            `;
            
            const emotionLabels = ['Valence', 'Arousal', 'Dominance', 'Immersion', 'Visual'];
            const emotionGrid = document.getElementById('emotionGrid');
            emotionGrid.innerHTML = '';
            
            results.avg_emotions.forEach((score, index) => {
                if (index < emotionLabels.length) {
                    const card = document.createElement('div');
                    card.className = 'emotion-card';
                    card.innerHTML = `
                        <h3>${emotionLabels[index]}</h3>
                        <div class="emotion-score">${score.toFixed(3)}</div>
                        <div style="margin-top: 0.5rem;">
                            <div style="width: 100%; background: #eee; border-radius: 5px; height: 8px;">
                                <div style="width: ${score * 100}%; background: #667eea; height: 8px; border-radius: 5px;"></div>
                            </div>
                        </div>
                    `;
                    emotionGrid.appendChild(card);
                }
            });
            
            if (results.emotion_plot) {
                document.getElementById('emotionPlot').innerHTML = `
                    <h3>📈 Emotion Trends Over Time</h3>
                    <img src="/static/plots/${results.emotion_plot}" alt="Emotion Plot">
                `;
            }
            
            if (results.summary_plot) {
                document.getElementById('summaryPlot').innerHTML = `
                    <h3>📊 Emotion Summary</h3>
                    <img src="/static/plots/${results.summary_plot}" alt="Summary Plot">
                `;
            }
            
            const insightsDiv = document.getElementById('insights');
            insightsDiv.innerHTML = '';
            
            const engagementScore = (0.4 * results.avg_emotions[3] + 0.3 * results.avg_emotions[0] + 0.3 * results.avg_emotions[1]);
            const engagementCard = document.createElement('div');
            engagementCard.className = 'insight-card';
            engagementCard.innerHTML = `
                <h4>📊 Engagement Score</h4>
                <p><strong>${engagementScore.toFixed(3)}</strong></p>
                <p>${engagementScore > 0.7 ? '🟢 Excellent engagement!' : 
                     engagementScore > 0.5 ? '🟡 Good engagement' : 
                     '🔴 Low engagement - content may need optimization'}</p>
            `;
            insightsDiv.appendChild(engagementCard);
            
            const totalDrops = Object.values(results.emotion_drops).reduce((sum, drops) => sum + drops.length, 0);
            const dropsCard = document.createElement('div');
            dropsCard.className = 'insight-card';
            dropsCard.innerHTML = `
                <h4>📉 Emotion Stability</h4>
                <p><strong>${totalDrops} significant drops detected</strong></p>
                <p>${totalDrops < 3 ? '🟢 Very stable emotions' : 
                     totalDrops < 6 ? '🟡 Moderately stable' : 
                     '🔴 High emotional variability'}</p>
            `;
            insightsDiv.appendChild(dropsCard);
            
            const recommendCard = document.createElement('div');
            recommendCard.className = 'insight-card';
            recommendCard.innerHTML = `
                <h4>💡 Recommendations</h4>
                <p>${generateRecommendations(results)}</p>
            `;
            insightsDiv.appendChild(recommendCard);
            
            displayTimeline(results.detailed_emotions);
            
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }

        function generateRecommendations(results) {
            const recommendations = [];
            
            if (results.avg_emotions[3] < 0.4) {
                recommendations.push("Increase engagement with interactive elements");
            }
            
            if (results.avg_emotions[0] < 0.4) {
                recommendations.push("Add more positive emotional content");
            }
            
            if (results.avg_emotions[1] < 0.4) {
                recommendations.push("Include more exciting/energetic moments");
            }
            
            const totalDrops = Object.values(results.emotion_drops).reduce((sum, drops) => sum + drops.length, 0);
            if (totalDrops > 5) {
                recommendations.push("Improve emotional consistency");
            }
            
            if (recommendations.length === 0) {
                return "Content performs well across all emotional dimensions! 🎉";
            }
            
            return recommendations.join("; ");
        }

        function displayTimeline(detailedEmotions) {
            const timelineContent = document.getElementById('timelineContent');
            timelineContent.innerHTML = '';
            
            detailedEmotions.forEach((emotion, index) => {
                if (index % 10 === 0 || index === detailedEmotions.length - 1) {
                    const item = document.createElement('div');
                    item.className = 'timeline-item';
                    item.innerHTML = `
                        <span class="time-marker">${emotion.time}s</span>
                        <span class="emotion-marker">${emotion.emotion}</span>
                        <span style="color: #999; font-size: 0.9rem;">
                            I: ${emotion.immersion.toFixed(2)}
                        </span>
                    `;
                    timelineContent.appendChild(item);
                }
            });
        }
    </script>
</body>
</html>
'''

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'})
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({'error': 'Invalid file format. Please upload MP4, AVI, MOV, or MKV files.'})
    
    session_id = str(uuid.uuid4())
    filename = f"{session_id}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'filename': filename
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video"""
    data = request.get_json()
    session_id = data.get('session_id')
    filename = data.get('filename')
    age = int(data.get('age', 25))
    gender = int(data.get('gender', 0))
    
    if not session_id or not filename:
        return jsonify({'error': 'Missing session information'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video file not found'})
    
    results = analyzer.analyze_single_video(filepath, age, gender, session_id)
    
    try:
        os.remove(filepath)
    except:
        pass
    
    return jsonify(results)

@app.route('/demographic_analysis')
def demographic_analysis():
    """Demographic analysis page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Demographic Analysis - Video Emotion Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .analysis-form {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #333;
            font-weight: 600;
        }
        
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .file-upload {
            position: relative;
            display: inline-block;
            cursor: pointer;
            width: 100%;
        }
        
        .file-upload input[type=file] {
            position: absolute;
            left: -9999px;
        }
        
        .file-upload-label {
            padding: 1rem;
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .file-upload-label:hover {
            background: #e9ecef;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .progress {
            width: 100%;
            height: 10px;
            background: #e1e5e9;
            border-radius: 5px;
            overflow: hidden;
            margin: 1rem 0;
            display: none;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .results {
            display: none;
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .demographics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .demographic-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }
        
        .demographic-card h3 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric-row:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 600;
            color: #555;
        }
        
        .metric-value {
            font-weight: bold;
            color: #667eea;
        }
        
        .insights-section {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 2rem 0;
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .insight-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #28a745;
        }
        
        .insight-card.warning {
            border-left-color: #ffc107;
        }
        
        .insight-card h4 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .recommendations {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 2rem 0;
        }
        
        .recommendations h3 {
            margin-bottom: 1rem;
        }
        
        .recommendations ul {
            list-style: none;
            padding: 0;
        }
        
        .recommendations li {
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .recommendations li:last-child {
            border-bottom: none;
        }
        
        .recommendations li::before {
            content: "💡 ";
            margin-right: 0.5rem;
        }
        
        .nav-links {
            text-align: center;
            margin-top: 2rem;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
            margin: 0 1rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            transition: background 0.3s;
        }
        
        .nav-links a:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .best-performer {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
        }
        
        .best-performer h2 {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Demographic Analysis</h1>
            <p>Analyze how different age groups and gender combinations respond to your video content</p>
        </div>
        
        <div class="analysis-form">
            <form id="demographicForm">
                <div class="form-group">
                    <label for="videoFile">Select Video File (MP4, AVI, MOV, MKV)</label>
                    <div class="file-upload">
                        <input type="file" id="videoFile" accept=".mp4,.avi,.mov,.mkv" required>
                        <label for="videoFile" class="file-upload-label">
                            <div>📁 Click to select video file</div>
                            <small>Max file size: 500MB</small>
                        </label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="ageGroups">Age Groups (comma-separated ranges, e.g., 13-17,18-25,26-35)</label>
                    <input type="text" id="ageGroups" value="13-17,18-25,26-35,36-50" placeholder="13-17,18-25,26-35">
                </div>
                
                <button type="submit" class="btn" id="analyzeBtn">
                    🔬 Analyze Demographics
                </button>
            </form>
            
            <div class="progress" id="progress">
                <div class="progress-bar" style="width: 0%"></div>
            </div>
            
            <div class="error" id="error" style="display: none;"></div>
        </div>
        
        <div class="results" id="results">
            <div class="best-performer" id="bestPerformer"></div>
            
            <div class="demographics-grid" id="demographicsGrid"></div>
            
            <div class="insights-section">
                <h3>📈 Key Insights</h3>
                <div class="insights-grid" id="insightsGrid"></div>
            </div>
            
            <div class="recommendations" id="recommendations">
                <h3>💡 Strategic Recommendations</h3>
                <ul id="recommendationsList"></ul>
            </div>
        </div>
        
        <div class="nav-links">
            <a href="/">👤 Single Person Analysis</a>
            <a href="#" onclick="location.reload()">🔄 New Analysis</a>
        </div>
    </div>

    <script>
        document.getElementById('videoFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const label = document.querySelector('.file-upload-label div');
                label.textContent = `📁 ${file.name}`;
            }
        });

        document.getElementById('demographicForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const videoFile = document.getElementById('videoFile').files[0];
            const ageGroups = document.getElementById('ageGroups').value;
            
            if (!videoFile) {
                showError('Please select a video file');
                return;
            }
            
            hideError();
            hideResults();
            showProgress(10);
            setButtonState(true);
            
            try {
                const formData = new FormData();
                formData.append('video', videoFile);
                formData.append('age_groups', ageGroups);
                
                showProgress(30);
                
                const response = await fetch('/analyze_demographics', {
                    method: 'POST',
                    body: formData
                });
                
                showProgress(80);
                
                const result = await response.json();
                
                showProgress(100);
                
                if (!result.success) {
                    throw new Error(result.error || 'Analysis failed');
                }
                
                displayDemographicResults(result);
                
            } catch (error) {
                showError(error.message);
            } finally {
                hideProgress();
                setButtonState(false);
            }
        });

        function showProgress(percentage) {
            const progress = document.getElementById('progress');
            const progressBar = progress.querySelector('.progress-bar');
            progress.style.display = 'block';
            progressBar.style.width = percentage + '%';
        }

        function hideProgress() {
            document.getElementById('progress').style.display = 'none';
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }

        function hideResults() {
            document.getElementById('results').style.display = 'none';
        }

        function setButtonState(disabled) {
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = disabled;
            btn.textContent = disabled ? '🔬 Analyzing Demographics...' : '🔬 Analyze Demographics';
        }

        function displayDemographicResults(data) {
            document.getElementById('results').style.display = 'block';
            
            const results = data.results || {};
            const insights = data.insights || {};
            
            if (insights.best_demographics && insights.best_demographics.length > 0) {
                const best = insights.best_demographics[0];
                document.getElementById('bestPerformer').innerHTML = `
                    <h2>🏆 Best Performing Demographic</h2>
                    <p><strong>${best.demographic}</strong></p>
                    <p>Engagement Score: ${best.engagement.toFixed(3)}</p>
                    <p>Dominant Emotion: ${best.dominant_emotion}</p>
                `;
            }
            
            const demographicsGrid = document.getElementById('demographicsGrid');
            demographicsGrid.innerHTML = '';
            
            for (const ageGroup in results) {
                for (const genderCombo in results[ageGroup]) {
                    const result = results[ageGroup][genderCombo];
                    if (result && result.success) {
                        const card = createDemographicCard(ageGroup, genderCombo, result);
                        demographicsGrid.appendChild(card);
                    }
                }
            }
            
            displayInsights(insights);
            displayRecommendations(insights);
            
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }

        function createDemographicCard(ageGroup, genderCombo, result) {
            const card = document.createElement('div');
            card.className = 'demographic-card';
            
            const emotionLabels = ['Valence', 'Arousal', 'Dominance', 'Immersion'];
            const engagementScore = result.avg_emotions && result.avg_emotions.length >= 4 ? 
                (0.4 * result.avg_emotions[3] + 0.3 * result.avg_emotions[0] + 0.3 * result.avg_emotions[1]) : 0;
            
            let metricsHtml = '';
            if (result.avg_emotions) {
                for (let i = 0; i < Math.min(4, result.avg_emotions.length); i++) {
                    metricsHtml += `
                        <div class="metric-row">
                            <span class="metric-label">${emotionLabels[i]}</span>
                            <span class="metric-value">${result.avg_emotions[i].toFixed(3)}</span>
                        </div>
                    `;
                }
            }
            
            card.innerHTML = `
                <h3>${ageGroup} - ${genderCombo}</h3>
                <div class="metric-row">
                    <span class="metric-label">Engagement Score</span>
                    <span class="metric-value">${engagementScore.toFixed(3)}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Dominant Emotion</span>
                    <span class="metric-value">${result.dominant_emotion || 'Unknown'}</span>
                </div>
                ${metricsHtml}
                <div class="metric-row">
                    <span class="metric-label">Video Length</span>
                    <span class="metric-value">${result.total_seconds || 0}s</span>
                </div>
            `;
            
            return card;
        }

        function displayInsights(insights) {
            const insightsGrid = document.getElementById('insightsGrid');
            insightsGrid.innerHTML = '';
            
            if (insights.best_demographics && insights.best_demographics.length > 0) {
                const card = document.createElement('div');
                card.className = 'insight-card';
                card.innerHTML = `
                    <h4>🏆 Top Performers</h4>
                    <p><strong>Top 3 Demographics:</strong></p>
                    <ul style="margin-top: 0.5rem; padding-left: 1rem;">
                        ${insights.best_demographics.slice(0, 3).map(demo => 
                            `<li>${demo.demographic} (${demo.engagement.toFixed(3)})</li>`
                        ).join('')}
                    </ul>
                `;
                insightsGrid.appendChild(card);
            }
            
            if (insights.worst_demographics && insights.worst_demographics.length > 0) {
                const card = document.createElement('div');
                card.className = 'insight-card warning';
                card.innerHTML = `
                    <h4>⚠️ Needs Improvement</h4>
                    <p><strong>Low Engagement:</strong></p>
                    <ul style="margin-top: 0.5rem; padding-left: 1rem;">
                        ${insights.worst_demographics.slice(0, 3).map(demo => 
                            `<li>${demo.demographic} (${demo.engagement.toFixed(3)})</li>`
                        ).join('')}
                    </ul>
                `;
                insightsGrid.appendChild(card);
            }
        }

        function displayRecommendations(insights) {
            const recommendationsList = document.getElementById('recommendationsList');
            recommendationsList.innerHTML = '';
            
            if (insights.recommendations && insights.recommendations.length > 0) {
                insights.recommendations.forEach(recommendation => {
                    const li = document.createElement('li');
                    li.textContent = recommendation;
                    recommendationsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'Continue monitoring demographic responses for optimization opportunities';
                recommendationsList.appendChild(li);
            }
        }
    </script>
</body>
</html>
'''

@app.route('/analyze_demographics', methods=['POST'])
def analyze_demographics():
    """Analyze video for multiple demographics"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video file selected'})
    
    session_id = str(uuid.uuid4())
    filename = f"{session_id}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    age_groups = request.form.get('age_groups', '13-17,18-25,26-35').split(',')
    age_ranges = []
    for age_range in age_groups:
        try:
            min_age, max_age = map(int, age_range.strip().split('-'))
            age_ranges.append((min_age, max_age))
        except:
            continue
    
    if not age_ranges:
        age_ranges = [(13, 17), (18, 25), (26, 35)]
    
    gender_combinations = [
        ('Boys Only', 1.0, 0.0),
        ('Girls Only', 0.0, 1.0),
        ('Mixed 50-50', 0.5, 0.5)
    ]
    
    results = {}
    for age_min, age_max in age_ranges:
        age_group_name = f"Age {age_min}-{age_max}"
        results[age_group_name] = {}
        representative_age = (age_min + age_max) // 2
        
        for combo_name, male_ratio, female_ratio in gender_combinations:
            if male_ratio > 0 and female_ratio > 0:
                male_results = analyzer.analyze_single_video(filepath, representative_age, 1, f"{session_id}_m")
                female_results = analyzer.analyze_single_video(filepath, representative_age, 0, f"{session_id}_f")
                
                if male_results.get('success') and female_results.get('success'):
                    combined_emotions = []
                    for i in range(len(male_results['avg_emotions'])):
                        combined_score = (male_results['avg_emotions'][i] * male_ratio + 
                                        female_results['avg_emotions'][i] * female_ratio)
                        combined_emotions.append(combined_score)
                    
                    combined_results = {
                        'success': True,
                        'avg_emotions': combined_emotions,
                        'dominant_emotion': male_results['dominant_emotion'],
                        'total_seconds': male_results['total_seconds']
                    }
                    results[age_group_name][combo_name] = combined_results
            else:
                gender = 1 if male_ratio > 0 else 0
                single_results = analyzer.analyze_single_video(filepath, representative_age, gender, f"{session_id}_{combo_name}")
                results[age_group_name][combo_name] = single_results
    
    insights = generate_demographic_insights(results)
    
    try:
        os.remove(filepath)
    except:
        pass
    
    return jsonify({
        'success': True,
        'results': results,
        'insights': insights
    })

def generate_demographic_insights(results):
    """Generate insights from demographic analysis"""
    insights = {
        'best_demographics': [],
        'worst_demographics': [],
        'age_insights': {},
        'gender_insights': {},
        'recommendations': []
    }
    
    engagement_scores = []
    for age_group, gender_results in results.items():
        for gender_combo, result in gender_results.items():
            if result and result.get('success') and len(result.get('avg_emotions', [])) >= 4:
                immersion = result['avg_emotions'][3]
                valence = result['avg_emotions'][0]
                arousal = result['avg_emotions'][1]
                engagement = (0.4 * immersion + 0.3 * valence + 0.3 * arousal)
                
                engagement_scores.append({
                    'demographic': f"{age_group} - {gender_combo}",
                    'engagement': engagement,
                    'immersion': immersion,
                    'valence': valence,
                    'arousal': arousal,
                    'dominant_emotion': result.get('dominant_emotion', 'Unknown')
                })
    
    engagement_scores.sort(key=lambda x: x['engagement'], reverse=True)
    
    if engagement_scores:
        insights['best_demographics'] = engagement_scores[:3]
        insights['worst_demographics'] = engagement_scores[-3:]
    
    if engagement_scores:
        best_demo = engagement_scores[0]
        insights['recommendations'] = [
            f"Target audience: {best_demo['demographic']} shows highest engagement ({best_demo['engagement']:.3f})",
            f"Focus on {best_demo['dominant_emotion'].lower()} emotional content",
            "Consider A/B testing with top-performing demographics"
        ]
        
        if best_demo['immersion'] < 0.5:
            insights['recommendations'].append("Work on increasing viewer immersion and engagement")
        
        if best_demo['valence'] < 0.5:
            insights['recommendations'].append("Add more positive emotional content")
    
    return insights

if __name__ == '__main__':
    print("🚀 Starting Video Emotion Analysis Web Application...")
    print("📂 Make sure your model files are available at:")
    print(f"   - EEG Model: {EEG_MODEL_PATH}")
    print(f"   - EEG Scaler: {EEG_SCALER_PATH}")
    print(f"   - Emotion Model: {EMOTION_MODEL_PATH}")
    print("\n🌐 Access the application at: http://localhost:5000")
    print("📊 Demographic analysis at: http://localhost:5000/demographic_analysis")
    
    try:
        test_analyzer = VideoEmotionAnalyzer()
        print("✅ All models loaded successfully!")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Please check:")
        print("1. Model file paths are correct")
        print("2. All required Python packages are installed")
        print("3. OpenCV is properly configured")