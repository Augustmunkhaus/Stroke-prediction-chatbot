import gradio as gr
import requests
import json
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from typing import Dict, Any, Tuple

# Sentiment analysis imports
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class StrokePredictionChatbot:
    def __init__(self, model_path=None, model_info_path=None):
        """
        Initialize the chatbot with the trained model and preprocessing components
        """
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.ollama_url = "http://localhost:11434/api/generate"

        # Add conversation memory
        self.user_profile = {}  # Stores the user's medical information
        self.conversation_context = []  # Stores conversation history

        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Default feature information (from your dataset)
        self.feature_info = {
            'age': {'type': 'numeric', 'range': (0, 120), 'unit': 'years'},
            'gender': {'type': 'categorical', 'values': ['Male', 'Female']},
            'hypertension': {'type': 'binary', 'values': [0, 1], 'description': '0: No, 1: Yes'},
            'heart_disease': {'type': 'binary', 'values': [0, 1], 'description': '0: No, 1: Yes'},
            'ever_married': {'type': 'categorical', 'values': ['Yes', 'No']},
            'work_type': {'type': 'categorical',
                          'values': ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']},
            'Residence_type': {'type': 'categorical', 'values': ['Urban', 'Rural']},
            'avg_glucose_level': {'type': 'numeric', 'range': (50, 300), 'unit': 'mg/dL'},
            'bmi': {'type': 'numeric', 'range': (10, 60), 'unit': 'kg/m²'},
            'smoking_status': {'type': 'categorical',
                               'values': ['formerly smoked', 'never smoked', 'smokes', 'Unknown']}
        }

        # Load model and preprocessing components if paths provided
        if model_path and model_info_path:
            self.load_model_components(model_path, model_info_path)
        else:
            print("WARNING: No model path provided. Using dummy predictions for testing.")

    def load_model_components(self, model_path, model_info_path):
        """Load the trained model and preprocessing components"""
        try:
            # Check if files exist
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return
            if not os.path.exists(model_info_path):
                print(f"Model info file not found: {model_info_path}")
                return

            # Load model
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model type: {type(self.model)}")

            # Load model info
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)

            print(f"Model info loaded: {model_info.keys()}")
            print(f"Sampling method: {model_info.get('sampling_method', 'Unknown')}")
            print(f"Best model: {model_info.get('best_model', 'Unknown')}")

            self.feature_names = model_info['feature_names']
            print(f"Feature names: {self.feature_names}")

            # Recreate label encoders
            for feature, classes in model_info['label_encoders'].items():
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                self.label_encoders[feature] = le

            # Recreate scaler
            self.scaler = StandardScaler()
            if model_info['scaler_mean'] and model_info['scaler_scale']:
                self.scaler.mean_ = np.array(model_info['scaler_mean'])
                self.scaler.scale_ = np.array(model_info['scaler_scale'])

            print("Model components loaded successfully!")

            # Test prediction with a known high-risk case
            print("\n=== TESTING MODEL WITH HIGH-RISK CASE ===")
            test_case = {
                'gender': 'Male', 'age': 70, 'hypertension': 1, 'heart_disease': 1,
                'ever_married': 'Yes', 'work_type': 'Private', 'Residence_type': 'Urban',
                'avg_glucose_level': 200, 'bmi': 35, 'smoking_status': 'smokes'
            }

            # Preprocess test case
            test_df = pd.DataFrame([test_case])

            # Encode categorical variables
            categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            for col in categorical_cols:
                if col in self.label_encoders:
                    test_df[col] = self.label_encoders[col].transform(test_df[col])

            # Reorder columns
            test_df = test_df.reindex(columns=self.feature_names, fill_value=0)

            # Scale features
            test_features = self.scaler.transform(test_df.values)

            # Make prediction
            test_prob = self.model.predict_proba(test_features)[0, 1]
            test_pred = self.model.predict(test_features)[0]

            print(f"Test case (70yr male, all risk factors): {test_prob:.4f} ({test_prob * 100:.1f}%)")
            print(f"Binary prediction: {test_pred}")

            if test_prob < 0.10:
                print("⚠️  WARNING: Model giving very low predictions even for high-risk cases!")
                print("This suggests the model might not be working correctly.")
            else:
                print("✅ Model seems to be working - high-risk case gets appropriate prediction")

        except Exception as e:
            print(f"Error loading model components: {e}")
            import traceback
            traceback.print_exc()
            print("Will use dummy model for demonstration")

    def call_ollama(self, prompt: str) -> str:
        """Call Ollama API to get LLM response"""
        try:
            payload = {
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.5
                }
            }

            response = requests.post(self.ollama_url, json=payload, timeout=30)

            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error calling Ollama: {response.status_code}"

        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"

    def extract_medical_info_improved(self, user_input: str) -> Dict[str, Any]:
        """
        Improved extraction - prioritize rule-based since LLM is failing
        """
        print(f"DEBUG - Input text: '{user_input}'")

        # Try rule-based extraction first since it's more reliable
        extracted_info = self.rule_based_extraction(user_input)

        # If rule-based didn't extract much, try LLM as backup
        if len(extracted_info) < 2:
            print("DEBUG - Rule-based found little, trying LLM...")
            try:
                llm_extracted = self.extract_with_llm(user_input)
                # Merge results, prioritizing rule-based
                for key, value in llm_extracted.items():
                    if key not in extracted_info:
                        extracted_info[key] = value
            except Exception as e:
                print(f"DEBUG - LLM extraction failed: {e}")

        print(f"DEBUG - Final extracted info: {extracted_info}")
        return extracted_info

    def extract_with_llm(self, user_input: str) -> Dict[str, Any]:
        """Backup LLM extraction method"""
        extraction_prompt = f"""
Extract medical information from: "{user_input}"

Rules:
- Age: look for "X years old", "X-year-old", "age X", "I'm X"
- Gender: "man"/"male"→Male, "woman"/"female"→Female  
- Blood pressure: "low BP"/"low blood pressure"→hypertension:0, "high BP"/"hypertension"→hypertension:1
- Smoking: "don't smoke"/"never smoked"→never smoked, "used to smoke"→formerly smoked, "I smoke"→smokes
- BMI: look for "BMI X" or "BMI is X"
- Work: "work from home"→Self-employed, "private sector"→Private, "government"→Govt_job
- Residence: "rural"→Rural, "urban"/"city"→Urban
- Marriage: "married"→Yes, "single"/"never married"→No

Return JSON only:
{{"age": number_or_null, "gender": "Male_or_Female_or_null", "hypertension": 0_or_1_or_null, "smoking_status": "value_or_null", "bmi": number_or_null, "work_type": "value_or_null", "Residence_type": "value_or_null", "ever_married": "Yes_or_No_or_null"}}"""

        llm_response = self.call_ollama(extraction_prompt)

        try:
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                extracted = json.loads(json_str)
                return self.validate_extracted_info(extracted)
        except:
            pass

        return {}

    def rule_based_extraction(self, user_input: str) -> Dict[str, Any]:
        """Much more robust rule-based extraction"""
        text = user_input.lower()
        extracted = {}

        print(f"DEBUG - Rule-based processing: '{text}'")

        # Age extraction - comprehensive patterns
        age_patterns = [
            r"(\d+)[-\s]*years?[-\s]*old",
            r"(\d+)[-\s]*year[-\s]*old",
            r"i['\s]*am['\s]*(\d+)",
            r"i['\s]*m['\s]*(\d+)",
            r"age[:\s]*(\d+)",
            r"(\d+)[-\s]*y/?o\b"
        ]

        for pattern in age_patterns:
            age_match = re.search(pattern, text)
            if age_match:
                age = int(age_match.group(1))
                if 0 <= age <= 120:
                    extracted['age'] = age
                    print(f"DEBUG - Found age: {age}")
                    break

        # Gender - improved detection with more patterns
        male_terms = ['man', 'male', 'guy', 'gentleman', 'mr', 'mister']
        female_terms = ['woman', 'female', 'lady', 'girl', 'mrs', 'miss', 'ms']

        if any(term in text for term in male_terms):
            extracted['gender'] = 'Male'
            print(f"DEBUG - Found gender: Male")
        elif any(term in text for term in female_terms):
            extracted['gender'] = 'Female'
            print(f"DEBUG - Found gender: Female")

        # BMI extraction - more patterns + overweight detection
        bmi_patterns = [
            r"bmi[:\s]*(?:is[:\s]*)?(\d+(?:\.\d+)?)",
            r"body mass index[:\s]*(?:is[:\s]*)?(\d+(?:\.\d+)?)",
            r"my bmi[:\s]*(?:is[:\s]*)?(\d+(?:\.\d+)?)"
        ]

        for pattern in bmi_patterns:
            bmi_match = re.search(pattern, text)
            if bmi_match:
                bmi = float(bmi_match.group(1))
                if 10 <= bmi <= 60:
                    extracted['bmi'] = bmi
                    print(f"DEBUG - Found BMI: {bmi}")
                    break

        # IMPROVED: Handle descriptive weight terms
        if 'overweight' in text or 'obese' in text or 'heavy' in text:
            if 'bmi' not in extracted:  # Only if no specific BMI found
                extracted['bmi'] = 32  # Assume moderately obese BMI
                print(f"DEBUG - Found descriptive weight (overweight/obese), setting BMI to 32")
        elif any(phrase in text for phrase in ['normal weight', 'healthy weight', 'not overweight', 'slim', 'thin']):
            if 'bmi' not in extracted:
                extracted['bmi'] = 24  # Normal BMI
                print(f"DEBUG - Found descriptive weight (normal), setting BMI to 24")

        # Blood pressure - FIXED LOGIC
        low_bp_phrases = ['low blood pressure', 'low bp', 'normal blood pressure', 'normal bp',
                          'have low blood pressure', 'i have low bp']
        high_bp_phrases = ['high blood pressure', 'high bp', 'hypertension', 'elevated blood pressure',
                           'have high blood pressure', 'i have high bp', 'have hypertension']

        # Check for explicit mentions
        if any(phrase in text for phrase in low_bp_phrases):
            extracted['hypertension'] = 0
            print(f"DEBUG - Found low/normal BP: hypertension = 0")
        elif any(phrase in text for phrase in high_bp_phrases):
            extracted['hypertension'] = 1
            print(f"DEBUG - Found high BP: hypertension = 1")

        # Heart disease detection
        heart_positive = ['heart disease', 'heart condition', 'cardiac disease', 'heart attack', 'have heart disease']
        heart_negative = ['no heart disease', 'healthy heart', 'no heart problems']

        if any(phrase in text for phrase in heart_positive):
            extracted['heart_disease'] = 1
            print(f"DEBUG - Found heart disease: 1")
        elif any(phrase in text for phrase in heart_negative):
            extracted['heart_disease'] = 0
            print(f"DEBUG - Found no heart disease: 0")

        # Smoking status - comprehensive patterns
        never_smoke_phrases = ["don't smoke", "never smoke", "non-smoke", "never smoked", "non smoker", "nonsmoker",
                               "do not smoke"]
        former_smoke_phrases = ['used to smoke', 'formerly smoke', 'quit smoke', 'ex-smoker', 'former smoker',
                                'stopped smoking', 'quit smoking']
        current_smoke_phrases = ['i smoke', ' smoke', 'smoker', 'smoking', 'current smoker']

        if any(phrase in text for phrase in never_smoke_phrases):
            extracted['smoking_status'] = 'never smoked'
            print(f"DEBUG - Found smoking: never smoked")
        elif any(phrase in text for phrase in former_smoke_phrases):
            extracted['smoking_status'] = 'formerly smoked'
            print(f"DEBUG - Found smoking: formerly smoked")
        elif any(phrase in text for phrase in current_smoke_phrases):
            extracted['smoking_status'] = 'smokes'
            print(f"DEBUG - Found smoking: smokes")

        # IMPROVED: Glucose level extraction with more patterns
        glucose_patterns = [
            r"glucose[:\s]*(?:is[:\s]*)?(?:around[:\s]*)?(\d+(?:\.\d+)?)",
            r"blood sugar[:\s]*(?:is[:\s]*)?(?:around[:\s]*)?(\d+(?:\.\d+)?)",
            r"sugar[:\s]*(?:level[:\s]*)?(?:is[:\s]*)?(?:around[:\s]*)?(\d+(?:\.\d+)?)"
        ]

        for pattern in glucose_patterns:
            glucose_match = re.search(pattern, text)
            if glucose_match:
                glucose = float(glucose_match.group(1))
                if 50 <= glucose <= 500:
                    extracted['avg_glucose_level'] = glucose
                    print(f"DEBUG - Found glucose: {glucose}")
                    break

        # IMPROVED: Handle descriptive glucose terms
        high_glucose_phrases = ['high blood sugar', 'high glucose', 'diabetes', 'diabetic', 'elevated glucose']
        normal_glucose_phrases = ['normal blood sugar', 'normal glucose', 'good blood sugar']

        if any(phrase in text for phrase in high_glucose_phrases):
            if 'avg_glucose_level' not in extracted:
                extracted['avg_glucose_level'] = 200  # High glucose
                print(f"DEBUG - Found descriptive glucose (high), setting to 200")
        elif any(phrase in text for phrase in normal_glucose_phrases):
            if 'avg_glucose_level' not in extracted:
                extracted['avg_glucose_level'] = 90  # Normal glucose
                print(f"DEBUG - Found descriptive glucose (normal), setting to 90")

        # Work type detection - improved
        work_patterns = {
            'Private': ['private sector', 'private company', 'work for company', 'corporate', 'company'],
            'Self-employed': ['self-employed', 'freelance', 'own business', 'work from home', 'self employed',
                              'freelancer'],
            'Govt_job': ['government', 'govt', 'public sector', 'government job', 'work for government',
                         'civil servant'],
            'Never_worked': ['never worked', 'unemployed', 'not working'],
            'children': ['student', 'school']
        }

        for work_type, keywords in work_patterns.items():
            if any(keyword in text for keyword in keywords):
                extracted['work_type'] = work_type
                print(f"DEBUG - Found work type: {work_type}")
                break

        # Residence type
        urban_terms = ['urban', 'city', 'metropolitan', 'downtown', 'town']
        rural_terms = ['rural', 'countryside', 'village', 'farm', 'rural area']

        if any(term in text for term in rural_terms):
            extracted['Residence_type'] = 'Rural'
            print(f"DEBUG - Found residence: Rural")
        elif any(term in text for term in urban_terms):
            extracted['Residence_type'] = 'Urban'
            print(f"DEBUG - Found residence: Urban")

        # Marriage status
        married_terms = ['married', 'wife', 'husband', 'spouse', 'am married']
        single_terms = ['single', 'never married', 'unmarried', 'not married']

        if any(term in text for term in married_terms):
            extracted['ever_married'] = 'Yes'
            print(f"DEBUG - Found marriage: Yes")
        elif any(term in text for term in single_terms):
            extracted['ever_married'] = 'No'
            print(f"DEBUG - Found marriage: No")

        print(f"DEBUG - Final rule-based extraction: {extracted}")
        return extracted

    def detect_hypothetical_question(self, user_input: str) -> bool:
        """Detect if the user is asking a hypothetical 'what if' question"""
        hypothetical_phrases = [
            'what if', 'suppose', 'imagine if', 'if i had', 'if my', 'what would happen',
            'how would', 'if i was', 'if i were', 'what about if', 'say i had'
        ]

        input_lower = user_input.lower()
        return any(phrase in input_lower for phrase in hypothetical_phrases)

    def extract_hypothetical_changes(self, user_input: str) -> Dict[str, Any]:
        """Extract what the user wants to change in their hypothetical scenario"""
        text = user_input.lower()
        changes = {}

        print(f"DEBUG - Extracting hypothetical changes from: '{text}'")

        # Gender changes
        if any(phrase in text for phrase in ['was female', 'was a woman', 'i was female', 'if i was a woman']):
            changes['gender'] = 'Female'
            print("DEBUG - Hypothetical: female")
        elif any(phrase in text for phrase in ['was male', 'was a man', 'i was male', 'if i was a man']):
            changes['gender'] = 'Male'
            print("DEBUG - Hypothetical: male")

        # Marriage status changes
        if any(phrase in text for phrase in ['was married', 'i was married', 'if i was married', 'had married']):
            changes['ever_married'] = 'Yes'
            print("DEBUG - Hypothetical: married")
        elif any(phrase in text for phrase in
                 ['was single', 'never married', 'i was single', 'if i was single', 'was unmarried']):
            changes['ever_married'] = 'No'
            print("DEBUG - Hypothetical: single/never married")

        # Work type changes
        if any(phrase in text for phrase in ['government job', 'govt job', 'worked for government', 'public sector']):
            changes['work_type'] = 'Govt_job'
            print("DEBUG - Hypothetical: government job")
        elif any(phrase in text for phrase in ['private sector', 'private company', 'worked for company']):
            changes['work_type'] = 'Private'
            print("DEBUG - Hypothetical: private sector")
        elif any(phrase in text for phrase in ['self-employed', 'freelance', 'own business', 'self employed']):
            changes['work_type'] = 'Self-employed'
            print("DEBUG - Hypothetical: self-employed")
        elif any(phrase in text for phrase in ['never worked', 'unemployed', 'not working']):
            changes['work_type'] = 'Never_worked'
            print("DEBUG - Hypothetical: never worked")
        elif any(phrase in text for phrase in ['student', 'in school']):
            changes['work_type'] = 'children'
            print("DEBUG - Hypothetical: student")

        # Residence type changes
        if any(phrase in text for phrase in ['lived in city', 'urban area', 'lived urban', 'in the city']):
            changes['Residence_type'] = 'Urban'
            print("DEBUG - Hypothetical: urban")
        elif any(phrase in text for phrase in ['lived rural', 'rural area', 'countryside', 'lived in village']):
            changes['Residence_type'] = 'Rural'
            print("DEBUG - Hypothetical: rural")

        # Blood pressure changes
        if any(phrase in text for phrase in ['high blood pressure', 'high bp', 'hypertension']):
            changes['hypertension'] = 1
            print("DEBUG - Hypothetical: high blood pressure")
        elif any(phrase in text for phrase in ['low blood pressure', 'low bp', 'normal bp', 'normal blood pressure']):
            changes['hypertension'] = 0
            print("DEBUG - Hypothetical: low blood pressure")

        # Smoking changes
        if any(phrase in text for phrase in ['smoked', 'smoking', 'i smoked', 'was smoking', 'if i smoked']):
            changes['smoking_status'] = 'smokes'
            print("DEBUG - Hypothetical: smoking")
        elif any(phrase in text for phrase in ['never smoked', "didn't smoke", 'non-smoker', 'no smoking']):
            changes['smoking_status'] = 'never smoked'
            print("DEBUG - Hypothetical: never smoked")
        elif any(phrase in text for phrase in ['used to smoke', 'formerly smoked', 'quit smoking']):
            changes['smoking_status'] = 'formerly smoked'
            print("DEBUG - Hypothetical: formerly smoked")

        # Heart disease changes
        if any(phrase in text for phrase in ['heart disease', 'heart condition', 'cardiac', 'had heart disease']):
            changes['heart_disease'] = 1
            print("DEBUG - Hypothetical: heart disease")
        elif any(
                phrase in text for phrase in ['no heart disease', 'no heart condition', 'healthy heart', 'no cardiac']):
            changes['heart_disease'] = 0
            print("DEBUG - Hypothetical: no heart disease")

        # Age changes
        age_match = re.search(r'(\d+)[-\s]*years?[-\s]*old', text)
        if not age_match:
            age_match = re.search(r'age[:\s]*(\d+)', text)
        if not age_match:
            age_match = re.search(r'was[:\s]*(\d+)', text)
        if age_match:
            changes['age'] = int(age_match.group(1))
            print(f"DEBUG - Hypothetical: age {changes['age']}")

        # BMI changes
        bmi_match = re.search(r'bmi[:\s]*(\d+(?:\.\d+)?)', text)
        if bmi_match:
            changes['bmi'] = float(bmi_match.group(1))
            print(f"DEBUG - Hypothetical: BMI {changes['bmi']}")

        # Weight status changes
        if any(phrase in text for phrase in ['overweight', 'obese', 'heavy']):
            changes['bmi'] = 30  # Assume obese BMI
            print("DEBUG - Hypothetical: overweight/obese")
        elif any(phrase in text for phrase in ['normal weight', 'healthy weight', 'not overweight']):
            changes['bmi'] = 24  # Normal BMI
            print("DEBUG - Hypothetical: normal weight")

        # Glucose/blood sugar changes
        glucose_patterns = [
            r'glucose[:\s]*(\d+(?:\.\d+)?)',
            r'blood sugar[:\s]*(\d+(?:\.\d+)?)',
            r'sugar[:\s]*(\d+(?:\.\d+)?)'
        ]

        for pattern in glucose_patterns:
            glucose_match = re.search(pattern, text)
            if glucose_match:
                glucose = float(glucose_match.group(1))
                if 50 <= glucose <= 500:
                    changes['avg_glucose_level'] = glucose
                    print(f"DEBUG - Hypothetical: glucose {glucose}")
                break

        # Blood sugar descriptive terms
        if any(phrase in text for phrase in ['low blood sugar', 'low glucose', 'normal blood sugar', 'normal glucose']):
            changes['avg_glucose_level'] = 90  # Normal glucose
            print("DEBUG - Hypothetical: low/normal blood sugar")
        elif any(phrase in text for phrase in ['high blood sugar', 'high glucose', 'diabetes', 'diabetic']):
            changes['avg_glucose_level'] = 200  # High glucose
            print("DEBUG - Hypothetical: high blood sugar")

        return changes

    def update_user_profile(self, extracted_info: Dict[str, Any], is_hypothetical: bool = False):
        """Update the stored user profile with new information"""
        if is_hypothetical:
            # Don't permanently update profile for hypothetical questions
            return

        for key, value in extracted_info.items():
            if value is not None:
                self.user_profile[key] = value
                print(f"DEBUG - Updated profile: {key} = {value}")

    def get_current_profile(self, extracted_info: Dict[str, Any], hypothetical_changes: Dict[str, Any] = None) -> Dict[
        str, Any]:
        """Get current profile, merging stored info with new extraction and hypothetical changes"""
        # Start with stored profile
        current_profile = self.user_profile.copy()

        # Update with any new extracted info
        for key, value in extracted_info.items():
            if value is not None:
                current_profile[key] = value

        # Apply hypothetical changes if any
        if hypothetical_changes:
            for key, value in hypothetical_changes.items():
                current_profile[key] = value
                print(f"DEBUG - Applied hypothetical change: {key} = {value}")

        # Fill missing values
        current_profile = self.fill_missing_values(current_profile)

        return current_profile

    def analyze_sentiment(self, text: str) -> Tuple[str, float, str]:
        """
        Analyze sentiment of user input using NLTK VADER
        Returns: (mood_category, confidence_score, emoji)
        """
        # Get sentiment scores
        scores = self.sentiment_analyzer.polarity_scores(text)

        # Extract compound score (overall sentiment)
        compound_score = scores['compound']

        # Categorize mood based on compound score
        if compound_score >= 0.05:
            mood = "😊 Good Mood"
            category = "good_mood"
        elif compound_score <= -0.05:
            mood = "😔 Bad Mood"
            category = "bad_mood"
        else:
            mood = "😐 Neutral"
            category = "neutral"

        # Convert to confidence percentage
        confidence = abs(compound_score) * 100

        print(f"DEBUG - Sentiment Analysis: '{text}' -> {category} (confidence: {confidence:.1f}%)")

        return mood, confidence, category

    def generate_hypothetical_response(self, user_input: str, stroke_prob: float, prediction: int,
                                       hypothetical_changes: Dict[str, Any], complete_info: Dict[str, Any]) -> str:
        """Generate response for hypothetical 'what if' scenarios"""

        # Create summary of changes and identify key factors
        changes_summary = []
        for key, value in hypothetical_changes.items():
            if key == 'hypertension':
                changes_summary.append("high blood pressure" if value == 1 else "normal blood pressure")
            elif key == 'smoking_status':
                if value == 'smokes':
                    changes_summary.append("smoking")
                else:
                    changes_summary.append(value.replace('_', ' '))
            elif key == 'heart_disease':
                changes_summary.append("heart disease" if value == 1 else "no heart disease")
            elif key == 'age':
                changes_summary.append(f"age {value}")
            elif key == 'bmi':
                changes_summary.append(f"BMI {value}")

        changes_text = " and ".join(changes_summary) if changes_summary else "those changes"

        # Simple response with risk and specific reasoning
        if stroke_prob < 0.05:
            response = f"**{stroke_prob:.1%} risk** - Very low. Even with {changes_text}, your other protective factors keep risk minimal."
        elif stroke_prob < 0.15:
            response = f"**{stroke_prob:.1%} risk** - Low to moderate. The {changes_text} would increase risk, but still manageable level."
        elif stroke_prob < 0.30:
            response = f"**{stroke_prob:.1%} risk** - Moderate. Having {changes_text} significantly increases risk due to these major factors."
        else:
            response = f"**{stroke_prob:.1%} risk** - High. The {changes_text} create substantial risk from multiple contributing factors."

        return response

    def validate_extracted_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted information"""
        validated = {}

        for key, value in info.items():
            if value is None or value == "null" or value == "":
                continue

            if key == 'age' and isinstance(value, (int, float)) and 0 <= value <= 120:
                validated[key] = int(value)
            elif key == 'bmi' and isinstance(value, (int, float)) and 10 <= value <= 60:
                validated[key] = float(value)
            elif key == 'avg_glucose_level' and isinstance(value, (int, float)) and 50 <= value <= 500:
                validated[key] = float(value)
            elif key in ['hypertension', 'heart_disease'] and isinstance(value, (int, bool)):
                validated[key] = int(value)
            elif key in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
                if isinstance(value, str) and value in self.feature_info[key]['values']:
                    validated[key] = value

        return validated

    def fill_missing_values(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing values with reasonable defaults"""
        defaults = {
            'age': 45,
            'gender': 'Male',
            'hypertension': 0,
            'heart_disease': 0,
            'ever_married': 'Yes',
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'avg_glucose_level': 95,
            'bmi': 25,
            'smoking_status': 'never smoked'
        }

        filled_info = {}
        for key, default_value in defaults.items():
            if key in extracted_info and extracted_info[key] is not None:
                filled_info[key] = extracted_info[key]
            else:
                filled_info[key] = default_value

        return filled_info

    def preprocess_input(self, medical_info: Dict[str, Any]) -> np.ndarray:
        """Preprocess the extracted information for model prediction"""
        try:
            print(f"DEBUG - Preprocessing input: {medical_info}")

            # Create a DataFrame with the input
            df = pd.DataFrame([medical_info])
            print(f"DEBUG - DataFrame created: {df}")
            print(f"DEBUG - DataFrame values before encoding: {df.values}")

            # Encode categorical variables
            categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

            for col in categorical_cols:
                if col in self.label_encoders and col in df.columns:
                    try:
                        original_value = df[col].iloc[0]
                        df[col] = self.label_encoders[col].transform(df[col])
                        print(f"DEBUG - Encoded {col}: {original_value} -> {df[col].iloc[0]}")
                        print(f"DEBUG - Available classes for {col}: {self.label_encoders[col].classes_}")
                    except ValueError as e:
                        print(f"DEBUG - Encoding error for {col}: {e}")
                        print(f"DEBUG - Available classes for {col}: {self.label_encoders[col].classes_}")
                        print(f"DEBUG - Trying to encode: {original_value}")
                        df[col] = 0

            print(f"DEBUG - DataFrame values after encoding: {df.values}")

            # Ensure correct column order
            if self.feature_names:
                df = df.reindex(columns=self.feature_names, fill_value=0)
                print(f"DEBUG - Reordered columns: {df.columns.tolist()}")
                print(f"DEBUG - Expected feature order: {self.feature_names}")
                print(f"DEBUG - Final feature values: {df.values[0]}")

            # Scale the features
            if self.scaler:
                print(f"DEBUG - Scaler mean: {self.scaler.mean_}")
                print(f"DEBUG - Scaler scale: {self.scaler.scale_}")
                features_scaled = self.scaler.transform(df.values)
                print(f"DEBUG - Scaled features: {features_scaled[0]}")
                print(f"DEBUG - Scaled features shape: {features_scaled.shape}")
            else:
                features_scaled = df.values
                print(f"DEBUG - No scaler, using raw features: {features_scaled[0]}")

            return features_scaled

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            # Return default values if preprocessing fails
            return np.array([[45, 1, 0, 0, 1, 0, 1, 95, 25, 1]])

    def predict_stroke_risk(self, features: np.ndarray) -> Tuple[float, float]:
        """Make stroke prediction using the trained model"""
        try:
            if self.model:
                print(f"DEBUG - Making prediction with features: {features}")
                # Get prediction probability
                stroke_probability = self.model.predict_proba(features)[0, 1]
                # Get binary prediction
                prediction = self.model.predict(features)[0]
                print(f"DEBUG - Prediction: {prediction}, Probability: {stroke_probability}")
                return float(stroke_probability), int(prediction)
            else:
                print("DEBUG - No model loaded, creating realistic dummy prediction")
                # Create more realistic dummy predictions based on actual risk factors
                # Assuming features are in order: [age, gender, hypertension, heart_disease, ever_married, work_type, residence, glucose, bmi, smoking]

                age = features[0][0] if len(features[0]) > 0 else 45
                gender = features[0][1] if len(features[0]) > 1 else 1  # 1 = Male typically
                hypertension = features[0][2] if len(features[0]) > 2 else 0
                heart_disease = features[0][3] if len(features[0]) > 3 else 0
                bmi_index = features[0][8] if len(features[0]) > 8 else 25  # Assuming BMI is at index 8

                print(
                    f"DEBUG - Dummy prediction factors: age={age}, gender={gender}, hypertension={hypertension}, heart_disease={heart_disease}")

                # More realistic risk calculation
                base_risk = 0.01  # 1% base risk

                # Age factor (major risk factor)
                if age < 30:
                    age_factor = 0.005
                elif age < 45:
                    age_factor = 0.01
                elif age < 55:
                    age_factor = 0.02
                elif age < 65:
                    age_factor = 0.04
                elif age < 75:
                    age_factor = 0.08
                else:
                    age_factor = 0.15

                # Hypertension factor
                hypertension_factor = 0.03 if hypertension else 0

                # Heart disease factor
                heart_factor = 0.05 if heart_disease else 0

                # Gender factor (males slightly higher risk)
                gender_factor = 0.01 if gender == 1 else 0

                total_risk = base_risk + age_factor + hypertension_factor + heart_factor + gender_factor
                total_risk = min(total_risk, 0.8)  # Cap at 80%

                prediction = 1 if total_risk > 0.1 else 0

                print(f"DEBUG - Calculated risk: {total_risk:.3f} ({total_risk * 100:.1f}%)")
                return total_risk, prediction

        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return 0.05, 0

    def generate_hybrid_response(self, user_input: str, stroke_prob: float, prediction: int,
                                 extracted_info: Dict[str, Any], complete_info: Dict[str, Any]) -> str:
        """Generate response using rule-based logic + LLM polish"""

        # STEP 1: Generate structured content with rules
        risk_percentage = f"{stroke_prob:.1%}"

        # Identify factors (regel-baseret)
        risk_factors = []
        protective_factors = []

        age = complete_info.get('age', 45)
        if age >= 65:
            risk_factors.append(f"age {age}")
        elif age < 35:
            protective_factors.append("young age")

        if complete_info.get('hypertension') == 1:
            risk_factors.append("high blood pressure")
        elif complete_info.get('hypertension') == 0:
            protective_factors.append("normal blood pressure")

        if complete_info.get('heart_disease') == 1:
            risk_factors.append("heart disease")

        smoking = complete_info.get('smoking_status')
        if smoking == 'smokes':
            risk_factors.append("current smoking")
        elif smoking == 'never smoked':
            protective_factors.append("never smoked")

        bmi = complete_info.get('bmi', 25)
        if bmi >= 30:
            risk_factors.append("obesity")
        elif bmi < 25:
            protective_factors.append("healthy weight")

        # STEP 2: Create base response (regel-baseret)
        if stroke_prob < 0.05:
            risk_level = "Very low"
            main_factors = " and ".join(protective_factors[:2]) if protective_factors else "no major risk factors"
        elif stroke_prob < 0.15:
            risk_level = "Low to moderate"
            main_factors = " and ".join(risk_factors[:2]) if risk_factors else "age and lifestyle factors"
        elif stroke_prob < 0.30:
            risk_level = "Moderate"
            main_factors = " and ".join(risk_factors[:3]) if risk_factors else "multiple factors"
        else:
            risk_level = "High"
            main_factors = " and ".join(risk_factors[:3]) if risk_factors else "multiple significant factors"

        # Base structured response
        base_response = f"**{risk_percentage} risk** - {risk_level}. Main factors: {main_factors}."

        # STEP 3: Polish with LLM
        polish_prompt = f"""
    Take this medical risk assessment and rewrite it to be more natural and empathetic while keeping the same information:

    ORIGINAL: "{base_response}"

    CONTEXT:
    - User input was: "{user_input}"
    - This is a stroke risk assessment
    - Risk factors present: {', '.join(risk_factors) if risk_factors else 'minimal'}
    - Protective factors: {', '.join(protective_factors) if protective_factors else 'standard'}

    INSTRUCTIONS:
    1. Keep the exact risk percentage and level: **{risk_percentage} risk** - {risk_level}
    2. Make the explanation more conversational and empathetic
    3. Keep it concise (under 30 words)
    4. Don't add medical advice
    5. Make it sound more natural than the template
    6. Dont write that this is a rewrite

    Rewrite:"""

        try:
            # Call LLM for polishing
            polished_response = self.call_ollama(polish_prompt)

            # Validate LLM response
            if len(polished_response) > 20 and risk_percentage in polished_response:
                return polished_response.strip()
            else:
                print("LLM polish failed, using base response")
                return base_response

        except Exception as e:
            print(f"LLM polish failed: {e}, using base response")
            return base_response

    def summarize_conversation(self, history) -> str:
        """Generate a summary of the conversation using LLM"""
        if not history:
            return "No conversation to summarize yet."

        # Build conversation text
        conversation_text = ""
        user_info_collected = []
        risk_assessments = []

        for user_msg, bot_msg in history:
            conversation_text += f"User: {user_msg}\nBot: {bot_msg}\n\n"

            # Extract risk percentages from bot messages
            import re
            risk_match = re.search(r'\*\*(\d+\.\d+%)\s*risk\*\*', bot_msg)
            if risk_match:
                risk_assessments.append(risk_match.group(1))

        # Create summary prompt
        summary_prompt = f"""
Summarize this stroke risk assessment conversation. Focus on:
1. User's medical profile (age, gender, health conditions, lifestyle)
2. Key risk factors discussed
3. Risk assessment results
4. Any hypothetical scenarios explored

Conversation:
{conversation_text}

Provide a concise, medical summary in 3-4 sentences.
"""

        try:
            # Try to get LLM summary
            llm_summary = self.call_ollama(summary_prompt)

            # If LLM fails or gives error, create rule-based summary
            if "Error" in llm_summary or len(llm_summary) < 50:
                return self.create_rule_based_summary(history)

            return llm_summary

        except Exception as e:
            print(f"LLM summary failed: {e}")
            return self.create_rule_based_summary(history)

    def create_rule_based_summary(self, history) -> str:
        """Create a rule-based summary when LLM is unavailable"""
        if not history:
            return "No conversation to summarize yet."

        # Extract information from user profile
        profile_summary = []
        if self.user_profile:
            age = self.user_profile.get('age')
            gender = self.user_profile.get('gender')
            if age and gender:
                profile_summary.append(f"{age}-year-old {gender.lower()}")

            conditions = []
            if self.user_profile.get('hypertension') == 1:
                conditions.append("high blood pressure")
            if self.user_profile.get('heart_disease') == 1:
                conditions.append("heart disease")
            if self.user_profile.get('smoking_status') == 'smokes':
                conditions.append("current smoker")
            elif self.user_profile.get('smoking_status') == 'formerly smoked':
                conditions.append("former smoker")

            if conditions:
                profile_summary.append(f"with {', '.join(conditions)}")

        # Extract latest risk assessment
        latest_risk = None
        for user_msg, bot_msg in reversed(history):
            risk_match = re.search(r'\*\*(\d+\.\d+%)\s*risk\*\*', bot_msg)
            if risk_match:
                latest_risk = risk_match.group(1)
                break

        # Count messages
        total_exchanges = len(history)

        # Build summary
        summary_parts = []

        if profile_summary:
            summary_parts.append(f"User profile: {' '.join(profile_summary)}.")

        if latest_risk:
            summary_parts.append(f"Current stroke risk assessment: {latest_risk}.")

        summary_parts.append(
            f"Conversation included {total_exchanges} exchanges about stroke risk factors and health assessment.")

        # Check for hypothetical questions
        hypothetical_count = sum(1 for user_msg, _ in history if self.detect_hypothetical_question(user_msg))
        if hypothetical_count > 0:
            summary_parts.append(f"Explored {hypothetical_count} hypothetical scenario(s).")

        return " ".join(summary_parts)

    def chat_response(self, user_input: str, history) -> Tuple[str, list, str]:
        """Main chatbot response function with sentiment analysis"""
        try:
            # Analyze sentiment first
            mood, confidence, category = self.analyze_sentiment(user_input)
            sentiment_display = f"{mood}\nConfidence: {confidence:.1f}%"

            # Check if the question is stroke-related
            stroke_keywords = [
                'age', 'years old', 'year old', 'yo', 'y/o', 'gender', 'male', 'female', 'man', 'woman',
                'hypertension', 'high blood pressure', 'hbp', 'blood pressure', 'bp', 'heart disease',
                'heart condition', 'heart attack', 'cardiac', 'married', 'single', 'ever married',
                'marital status', 'work', 'job', 'employed', 'self-employed', 'government job', 'private',
                'residence', 'urban', 'rural', 'city', 'countryside', 'glucose', 'blood sugar', 'diabetes',
                'diabetic', 'bmi', 'body mass', 'weight', 'obese', 'overweight', 'smoke', 'smoking',
                'smoker', 'non-smoker', 'tobacco', 'stroke', 'brain attack', 'cva', 'risk factor',
                'numbness', 'weakness', 'slurred speech', 'vision problem', 'dizziness',
                'what if', 'suppose', 'imagine', 'how would', 'what about',  # hypothetical keywords
                'explain', 'why', 'how', 'risk', 'percent', '%'  # follow-up question keywords
            ]

            input_lower = user_input.lower()
            is_stroke_related = any(keyword in input_lower for keyword in stroke_keywords)

            if not is_stroke_related:
                response = "I'm a stroke risk assessment chatbot. I can help you understand your stroke risk based on factors like age, blood pressure, heart health, BMI, smoking status, and other health information. Please tell me about your health profile!"
                history.append([user_input, response])
                return "", history, sentiment_display

            # Detect if this is a hypothetical question
            is_hypothetical = self.detect_hypothetical_question(user_input)
            print(f"DEBUG - Is hypothetical: {is_hypothetical}")

            if is_hypothetical and self.user_profile:
                # Extract hypothetical changes
                hypothetical_changes = self.extract_hypothetical_changes(user_input)
                extracted_info = {}  # Don't extract new base info for hypotheticals
                complete_info = self.get_current_profile(extracted_info, hypothetical_changes)

                print(f"DEBUG - Hypothetical scenario with changes: {hypothetical_changes}")
                print(f"DEBUG - Complete hypothetical profile: {complete_info}")

            else:
                # Regular extraction and profile update
                extracted_info = self.extract_medical_info_improved(user_input)
                print(f"DEBUG - Extracted info: {extracted_info}")

                # Update user profile with new information
                self.update_user_profile(extracted_info, is_hypothetical)

                # Get complete profile
                complete_info = self.get_current_profile(extracted_info)
                print(f"DEBUG - Complete profile: {complete_info}")

            # Make prediction
            features = self.preprocess_input(complete_info)
            stroke_prob, prediction = self.predict_stroke_risk(features)

            # Determine risk level
            if stroke_prob < 0.05:
                risk_level = "Very Low"
            elif stroke_prob < 0.15:
                risk_level = "Low"
            elif stroke_prob < 0.30:
                risk_level = "Moderate"
            else:
                risk_level = "Elevated"

            # Generate response
            if is_hypothetical:
                bot_response = self.generate_hypothetical_response(
                    user_input, stroke_prob, prediction, hypothetical_changes, complete_info
                )
            else:
                bot_response = self.generate_hybrid_response(
                    user_input, stroke_prob, prediction, extracted_info, complete_info
                )

            # Just return the bot response without technical analysis
            history.append([user_input, bot_response])
            return "", history, sentiment_display

        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your health information."
            history.append([user_input, error_response])
            return "", history, sentiment_display


def create_interface():
    """Create and launch the Gradio interface"""
    # Initialize chatbot with best stroke model
    MODEL_PATH = "C:/Users/August/PycharmProjects/stroke_prediction/Model/best_stroke_model_moderate_weights.pkl"
    MODEL_INFO_PATH = "C:/Users/August/PycharmProjects/stroke_prediction/Model/model_info_moderate_weights.json"

    # Check if model files exist
    if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_INFO_PATH):
        print(f"Loading BALANCED WEIGHTS model from {MODEL_PATH}")
        chatbot = StrokePredictionChatbot(MODEL_PATH, MODEL_INFO_PATH)
    else:
        print(f"Balanced weights model files not found:")
        print(f"Looking for: {MODEL_PATH}")
        print(f"Looking for: {MODEL_INFO_PATH}")
        print("Using demo mode with dummy predictions.")
        chatbot = StrokePredictionChatbot()

    # Create Gradio interface
    with gr.Blocks(title="Stroke Risk Assessment Chatbot", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🧠 Stroke Risk Assessment Chatbot (Balanced Weights Model)

        I'll help assess your stroke risk based on your health information. 

        **Just tell me about yourself naturally!** For example:
        - "I'm a 26-year-old man, I don't smoke, I have low blood pressure, my BMI is 23"
        - "I'm 65, male, I smoke, have high blood pressure, BMI around 28"
        - "45-year-old female, never married, work in government, don't smoke, live in the city"

        **⚠️ Important:** This is for educational purposes only and should not replace professional medical advice.

        **🔬 Model Info:** Using Logistic Regression with Balanced Class Weights (91.4% accuracy)
        """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot_interface = gr.Chatbot(
                    height=500,
                    placeholder="Ask me about stroke risk factors...",
                    show_label=False
                )

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Tell me about your age, health conditions, lifestyle...",
                        lines=2,
                        scale=4,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")

                gr.Examples(
                    examples=[
                        "I'm a 26-year-old man, I don't smoke, I have low blood pressure, my BMI is 23",
                        "I'm a 60-year-old male, I have hypertension, used to smoke, work in private sector, BMI around 27",
                        "I'm 45 years old, female, never smoked, married, work from home, live in rural area",
                        "I'm 70, have heart disease, my blood sugar is around 180, I'm overweight"
                    ],
                    inputs=user_input
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🎭 Mood Tracker")
                sentiment_display = gr.Textbox(
                    label="Current Mood",
                    value="😐 Neutral\nConfidence: 0.0%",
                    interactive=False,
                    lines=3,
                    max_lines=3
                )

                gr.Markdown("### 📋 Conversation Summary")

                with gr.Row():
                    summary_btn = gr.Button("Generate Summary", variant="secondary")

                summary_display = gr.Textbox(
                    label="Summary",
                    value="Click 'Generate Summary' to summarize your conversation...",
                    interactive=False,
                    lines=6,
                    max_lines=8
                )

        # Event handlers
        def update_summary(history):
            summary = chatbot.summarize_conversation(history)
            return summary

        submit_btn.click(
            chatbot.chat_response,
            inputs=[user_input, chatbot_interface],
            outputs=[user_input, chatbot_interface, sentiment_display]
        )

        user_input.submit(
            chatbot.chat_response,
            inputs=[user_input, chatbot_interface],
            outputs=[user_input, chatbot_interface, sentiment_display]
        )

        summary_btn.click(
            update_summary,
            inputs=[chatbot_interface],
            outputs=[summary_display]
        )

    return interface


if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )