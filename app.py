# Credit to Claude 4o for frontend UI code
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import os
import torch
import yaml
from werkzeug.utils import secure_filename
import tempfile
import shutil
import glob
import json

from e2e_model_transMIL.training.inference import load_inference_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'.svs', '.ndpi', '.tif', '.tiff'}

# WSI directories on the host
WSI_BASE_DIR = "/data/ar2221/all_wsis"
WSI_SETS = ["wsi_set1", "wsi_set2", "wsi_set3"]

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and config
MODEL_PATH = "e2e_model_transMIL/checkpoints_transmil_final/fold_4_best.pth"
CONFIG_PATH = "e2e_model_transMIL/configs/transmil_regressor_config.yaml"

# Initialize model
inference_model = None

def allowed_file(filename):
    return Path(filename).suffix.lower() in app.config['ALLOWED_EXTENSIONS']

def get_available_slides():
    """Get all available slides from the WSI directories"""
    slides = {}
    for wsi_set in WSI_SETS:
        wsi_dir = os.path.join(WSI_BASE_DIR, wsi_set)
        if os.path.exists(wsi_dir):
            slide_files = []
            for ext in ['.svs', '.ndpi', '.tif', '.tiff']:
                slide_files.extend(glob.glob(os.path.join(wsi_dir, f"*{ext}")))
            slides[wsi_set] = [os.path.basename(f) for f in slide_files]
    return slides

@app.route('/')
def index():
    slides = get_available_slides()
    return render_template('index.html', available_slides=slides)

@app.route('/api/slides')
def get_slides():
    """API endpoint to get available slides"""
    slides = get_available_slides()
    return jsonify(slides)

@app.route('/process_slide', methods=['POST'])
def process_slide():
    """Process a slide selected from the dropdown"""
    data = request.get_json()
    if not data or 'slide_path' not in data:
        return jsonify({'error': 'No slide path provided'}), 400
    
    slide_path = data['slide_path']
    enable_visualizations = data.get('enable_visualizations', True)
    
    # Validate the slide path
    if not slide_path.startswith(WSI_BASE_DIR):
        return jsonify({'error': 'Invalid slide path'}), 400
    
    if not os.path.exists(slide_path):
        return jsonify({'error': 'Slide file not found'}), 404
    
    try:
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process the WSI
            results = inference_model.predict_single_wsi(
                slide_path,
                output_dir=temp_dir,
                save_visualizations=enable_visualizations
            )
            
            # Copy files to a permanent location
            slide_name = Path(slide_path).stem
            output_dir = os.path.join(app.config['UPLOAD_FOLDER'], slide_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy files based on visualization setting
            viz_files = []
            if enable_visualizations:
                # Copy all files including visualizations
                for file in Path(temp_dir).glob('**/*'):
                    if file.is_file():
                        shutil.copy2(file, output_dir)
                
                # Get list of visualization files
                for file in Path(output_dir).glob('*.png'):
                    viz_files.append(f'/visualizations/{slide_name}/{file.name}')
            else:
                # Only copy non-visualization files (like inference summary)
                for file in Path(temp_dir).glob('**/*'):
                    if file.is_file() and not file.name.endswith('.png'):
                        shutil.copy2(file, output_dir)
            
            # Extract prediction from results - handle different possible formats
            prediction_score = None
            if 'predicted_scores' in results:
                raw_score = float(results['predicted_scores'][0])
                # Safeguard: clamp to valid Banff scale range (0-3)
                prediction_score = max(0.0, min(3.0, raw_score))
                if raw_score != prediction_score:
                    app.logger.warning(f"Model output {raw_score} was clamped to {prediction_score}")
            elif 'predictions' in results:
                raw_score = float(results['predictions'][0])
                prediction_score = max(0.0, min(3.0, raw_score))
                if raw_score != prediction_score:
                    app.logger.warning(f"Model predictions output {raw_score} was clamped to {prediction_score}")
            elif 'prediction' in results:
                raw_score = float(results['prediction'])
                prediction_score = max(0.0, min(3.0, raw_score))
                if raw_score != prediction_score:
                    app.logger.warning(f"Model prediction output {raw_score} was clamped to {prediction_score}")
            else:
                # Try to read from inference summary file
                summary_file = os.path.join(output_dir, 'inference_summary.json')
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        if 'predicted_tubulitis_score' in summary:
                            raw_score = float(summary['predicted_tubulitis_score'])
                            prediction_score = max(0.0, min(3.0, raw_score))
                            if raw_score != prediction_score:
                                app.logger.warning(f"Summary predicted_tubulitis_score {raw_score} was clamped to {prediction_score}")
                        elif 'prediction' in summary:
                            raw_score = float(summary['prediction'])
                            prediction_score = max(0.0, min(3.0, raw_score))
                            if raw_score != prediction_score:
                                app.logger.warning(f"Summary prediction {raw_score} was clamped to {prediction_score}")
                
            if prediction_score is None:
                # Fallback: look for any numeric value in the results
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        raw_score = float(value)
                        prediction_score = max(0.0, min(3.0, raw_score))
                        if raw_score != prediction_score:
                            app.logger.warning(f"Fallback numeric value {raw_score} was clamped to {prediction_score}")
                        break
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        try:
                            raw_score = float(value[0])
                            prediction_score = max(0.0, min(3.0, raw_score))
                            if raw_score != prediction_score:
                                app.logger.warning(f"Fallback list value {raw_score} was clamped to {prediction_score}")
                            break
                        except (ValueError, TypeError):
                            continue
            
            if prediction_score is None:
                return jsonify({'error': 'Could not extract prediction score from model results'}), 500
            
            return jsonify({
                'success': True,
                'prediction': prediction_score,
                'slide_name': slide_name,
                'visualization_files': viz_files
            })
            
    except Exception as e:
        app.logger.error(f"Error processing slide {slide_path}: {str(e)}")
        return jsonify({'error': f'Error processing slide: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Get visualization setting from form data
    enable_visualizations = request.form.get('enable_visualizations', 'true').lower() == 'true'
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        try:
            # Process the WSI
            results = inference_model.predict_single_wsi(
                file_path,
                output_dir=temp_dir,
                save_visualizations=enable_visualizations
            )
            
            # Copy files to a permanent location
            output_dir = os.path.join(app.config['UPLOAD_FOLDER'], Path(filename).stem)
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy files based on visualization setting
            viz_files = []
            if enable_visualizations:
                # Copy all files including visualizations
                for file in Path(temp_dir).glob('**/*'):
                    if file.is_file():
                        shutil.copy2(file, output_dir)
                
                # Get list of visualization files
                for file in Path(output_dir).glob('*.png'):
                    viz_files.append(f'/visualizations/{Path(filename).stem}/{file.name}')
            else:
                # Only copy non-visualization files (like inference summary)
                for file in Path(temp_dir).glob('**/*'):
                    if file.is_file() and not file.name.endswith('.png'):
                        shutil.copy2(file, output_dir)
            
            # Extract prediction from results - handle different possible formats
            prediction_score = None
            if 'predicted_scores' in results:
                raw_score = float(results['predicted_scores'][0])
                # Safeguard: clamp to valid Banff scale range (0-3)
                prediction_score = max(0.0, min(3.0, raw_score))
                if raw_score != prediction_score:
                    app.logger.warning(f"Model output {raw_score} was clamped to {prediction_score}")
            elif 'predictions' in results:
                raw_score = float(results['predictions'][0])
                prediction_score = max(0.0, min(3.0, raw_score))
                if raw_score != prediction_score:
                    app.logger.warning(f"Model predictions output {raw_score} was clamped to {prediction_score}")
            elif 'prediction' in results:
                raw_score = float(results['prediction'])
                prediction_score = max(0.0, min(3.0, raw_score))
                if raw_score != prediction_score:
                    app.logger.warning(f"Model prediction output {raw_score} was clamped to {prediction_score}")
            else:
                # Try to read from inference summary file
                summary_file = os.path.join(output_dir, 'inference_summary.json')
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        if 'predicted_tubulitis_score' in summary:
                            raw_score = float(summary['predicted_tubulitis_score'])
                            prediction_score = max(0.0, min(3.0, raw_score))
                            if raw_score != prediction_score:
                                app.logger.warning(f"Summary predicted_tubulitis_score {raw_score} was clamped to {prediction_score}")
                        elif 'prediction' in summary:
                            raw_score = float(summary['prediction'])
                            prediction_score = max(0.0, min(3.0, raw_score))
                            if raw_score != prediction_score:
                                app.logger.warning(f"Summary prediction {raw_score} was clamped to {prediction_score}")
                
            if prediction_score is None:
                # Fallback: look for any numeric value in the results
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        raw_score = float(value)
                        prediction_score = max(0.0, min(3.0, raw_score))
                        if raw_score != prediction_score:
                            app.logger.warning(f"Fallback numeric value {raw_score} was clamped to {prediction_score}")
                        break
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        try:
                            raw_score = float(value[0])
                            prediction_score = max(0.0, min(3.0, raw_score))
                            if raw_score != prediction_score:
                                app.logger.warning(f"Fallback list value {raw_score} was clamped to {prediction_score}")
                            break
                        except (ValueError, TypeError):
                            continue
            
            if prediction_score is None:
                return jsonify({'error': 'Could not extract prediction score from model results'}), 500
            
            return jsonify({
                'success': True,
                'prediction': prediction_score,
                'slide_name': Path(filename).stem,
                'visualization_files': viz_files
            })
            
        except Exception as e:
            app.logger.error(f"Error processing uploaded file {filename}: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/visualizations/<path:filename>')
def get_visualization(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def init_model():
    global inference_model
    try:
        inference_model = load_inference_model(MODEL_PATH, CONFIG_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == '__main__':
    init_model()
    # Modified to listen on all interfaces
    app.run(host='0.0.0.0', port=5000, debug=False) 