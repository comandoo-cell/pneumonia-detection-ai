import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
from gradcam import generate_gradcam
from datetime import datetime
import database as db
from pdf_generator import generate_pdf_report, generate_report_filename

model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '..', 'best_model_NEW_TEST.h5'))

UPLOAD_FOLDER = 'static/uploads'
HEATMAP_FOLDER = 'static/heatmaps'
REPORTS_FOLDER = 'static/reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

db.init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            result = 'PNEUMONIA'
            confidence_percent = confidence * 100
        else:
            result = 'NORMAL'
            confidence_percent = (1 - confidence) * 100
        
        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
        
        img_for_gradcam = image.load_img(file_path, target_size=(224, 224))
        img_array_gradcam = image.img_to_array(img_for_gradcam)
        img_array_gradcam = np.expand_dims(img_array_gradcam, axis=0)
        img_array_gradcam = preprocess_input(img_array_gradcam)
        
        gradcam_success = generate_gradcam(file_path, img_array_gradcam, model, heatmap_path)
        
        patient_name = request.form.get('patient_name', 'مجهول')
        patient_age = request.form.get('patient_age', None)
        patient_gender = request.form.get('patient_gender', 'غير محدد')
        patient_phone = request.form.get('patient_phone', '')
        tc_kimlik = request.form.get('tc_kimlik', '')
        notes = request.form.get('notes', '')
        
        if patient_name != 'مجهول':
            patient_id = db.add_patient(patient_name, patient_age, patient_gender, patient_phone, tc_kimlik)
        else:
            patient_id = None
        
        scan_id = db.add_scan(
            patient_id, 
            filename, 
            heatmap_filename if gradcam_success else None,
            result, 
            round(confidence_percent, 2),
            notes
        )
        
        return render_template('result.html', 
                             prediction=result, 
                             filename=filename,
                             confidence=round(confidence_percent, 2),
                             heatmap_filename=heatmap_filename if gradcam_success else None,
                             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             scan_id=scan_id)
    
    return redirect(request.url)

@app.route('/dashboard')
def dashboard():
    stats = db.get_statistics()
    recent_scans = db.get_all_scans(limit=10)
    recent_patients = db.get_recent_patients(limit=5)
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_scans=recent_scans,
                         recent_patients=recent_patients)

@app.route('/api/scans')
def api_scans():
    limit = request.args.get('limit', 50, type=int)
    scans = db.get_all_scans(limit=limit)
    return jsonify(scans)

@app.route('/api/statistics')
def api_statistics():
    stats = db.get_statistics()
    return jsonify(stats)

@app.route('/history')
def history():
    search_term = request.args.get('search', '')
    prediction_filter = request.args.get('filter', 'ALL')
    sort_by = request.args.get('sort', 'date')
    sort_order = request.args.get('order', 'desc')
    
    scans = db.search_and_filter_scans(search_term, prediction_filter, sort_by, sort_order)
    
    return render_template('history.html', scans=scans, 
                         search_term=search_term, 
                         prediction_filter=prediction_filter,
                         sort_by=sort_by,
                         sort_order=sort_order)

@app.route('/api/scans_timeline')
def scans_timeline():
    days = request.args.get('days', 30, type=int)
    data = db.get_scans_by_date_range(days)
    return jsonify(data)

@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
def delete_scan(scan_id):
    success = db.delete_scan(scan_id)
    if success:
        return jsonify({'success': True, 'message': 'Kayıt başarıyla silindi'})
    else:
        return jsonify({'success': False, 'message': 'Kayıt silinemedi'}), 400

@app.route('/generate_report/<int:scan_id>')
def generate_report(scan_id):
    scans = db.get_all_scans(limit=1000)
    scan = next((s for s in scans if s['id'] == scan_id), None)
    
    if not scan:
        return "Scan not found", 404
    
    report_data = {
        'scan_id': scan_id,
        'patient_name': scan.get('patient_name', 'غير محدد'),
        'age': scan.get('age', 'غير محدد'),
        'gender': scan.get('gender', 'غير محدد'),
        'tc_kimlik': scan.get('tc_kimlik', 'Belirtilmemis'),
        'phone': scan.get('phone', 'غير محدد'),
        'scan_date': scan['scan_date'],
        'prediction': scan['prediction'],
        'confidence': scan['confidence'],
        'notes': scan.get('notes', ''),
        'image_path': os.path.join(app.config['UPLOAD_FOLDER'], scan['image_filename']),
        'heatmap_path': os.path.join(app.config['HEATMAP_FOLDER'], scan['heatmap_filename']) if scan.get('heatmap_filename') else None
    }
    
    pdf_filename = generate_report_filename(scan_id)
    pdf_path = os.path.join(app.config['REPORTS_FOLDER'], pdf_filename)
    
    success = generate_pdf_report(report_data, pdf_path)
    
    if success:
        from flask import send_file
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
    else:
        return "Error generating report", 500

if __name__ == '__main__':
    url = 'http://127.0.0.1:5000/'
    print(f'\n\u25B6 افتح المتصفح: {url}\n')
    app.run(debug=True)
