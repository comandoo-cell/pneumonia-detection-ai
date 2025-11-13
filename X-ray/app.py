import os
import re
from datetime import datetime
from functools import wraps
from urllib.parse import urljoin, urlparse

import numpy as np
import tensorflow as tf
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.preprocessing import image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import database as db
from gradcam import generate_gradcam
from pdf_generator import generate_pdf_report, generate_report_filename

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "best_model_STRONG.h5")
MODEL_THRESHOLD = 0.45
MODEL_IMG_SIZE = (300, 300)
PREPROCESS = efficientnet_v2.preprocess_input
GRADCAM_LAYER_NAME = None
GRADCAM_LAYER_CANDIDATES = [
    "block6f_project_conv",
    "block5f_project_conv",
    "block5e_project_conv",
]
GRADCAM_MIN_INTENSITY = 0.06
GRADCAM_APPLY_THRESHOLD = True
GRADCAM_ALPHA = 0.22
GRADCAM_BLUR_KERNEL = 5

model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HEATMAP_FOLDER = os.path.join(BASE_DIR, 'static', 'heatmaps')
REPORTS_FOLDER = os.path.join(BASE_DIR, 'static', 'reports')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'change-me')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

db.init_db()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if 'doctor_id' not in session:
            flash('Devam etmek için giriş yapın.', 'warning')
            next_url = request.url if request.method == 'GET' else url_for('index')
            return redirect(url_for('login', next=next_url))
        return view_func(*args, **kwargs)

    return wrapped_view


@app.before_request
def load_logged_in_doctor():
    doctor_id = session.get('doctor_id')
    g.doctor = db.get_doctor_by_id(doctor_id) if doctor_id else None


@app.context_processor
def inject_current_doctor():
    return {'current_doctor': getattr(g, 'doctor', None)}


def is_safe_url(target):
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_username(username):
    if not username or len(username) < 3:
        return False, "Kullanıcı adı en az 3 karakter olmalıdır."
    
    if not re.search(r'[a-zA-ZğüşıöçĞÜŞİÖÇ]', username):
        return False, "Kullanıcı adı en az bir harf içermelidir."
    
    if not re.match(r'^[a-zA-Z0-9_ğüşıöçĞÜŞİÖÇ]+$', username):
        return False, "Kullanıcı adı sadece harf, rakam ve alt çizgi içerebilir."
    
    return True, ""


def validate_patient_name(name):
    if not name or len(name.strip()) < 2:
        return False, "Hasta adı en az 2 karakter olmalıdır."
    
    if not re.match(r'^[a-zA-ZğüşıöçĞÜŞİÖÇ\s]+$', name):
        return False, "Hasta adı sadece harf içermelidir (rakam kullanılamaz)."
    
    return True, ""


def validate_tc_kimlik(tc):
    if not tc:
        return True, ""
    
    tc = tc.strip()
    
    if not re.match(r'^\d{11}$', tc):
        return False, "TC Kimlik No 11 haneli rakam olmalıdır."
    
    return True, ""


def validate_phone(phone):
    if not phone:
        return True, ""
    
    phone_clean = re.sub(r'[\s\-\(\)]', '', phone)
    
    phone_clean = re.sub(r'^(\+90|0090|90|0)', '', phone_clean)
    
    if not re.match(r'^\d{10}$', phone_clean):
        return False, "Telefon numarası 10 haneli rakam olmalıdır (örn: 5XX XXX XX XX)."
    
    formatted_phone = f"+90{phone_clean}"
    
    return True, formatted_phone


@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('doctor_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        doctor = db.get_doctor_by_username(username)

        if doctor and check_password_hash(doctor['password_hash'], password):
            session['doctor_id'] = doctor['id']
            flash('Hoş geldiniz.', 'success')
            next_page = request.args.get('next')
            if next_page and not is_safe_url(next_page):
                next_page = None
            return redirect(next_page or url_for('index'))

        flash('Kullanıcı adı veya şifre hatalı.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('doctor_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        hospital_name = request.form.get('hospital_name', '').strip() or None
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not name or not username or not password:
            flash('Lütfen tüm zorunlu alanları doldurun.', 'warning')
        elif password != confirm_password:
            flash('Şifreler eşleşmiyor.', 'danger')
        else:
            is_valid, error_msg = validate_username(username)
            if not is_valid:
                flash(error_msg, 'danger')
            elif db.get_doctor_by_username(username):
                flash('Bu kullanıcı adı zaten kayıtlı.', 'danger')
            else:
                password_hash = generate_password_hash(password)
                doctor_id = db.add_doctor(name, username, password_hash, hospital_name)
                session['doctor_id'] = doctor_id
                flash('Hesabınız oluşturuldu.', 'success')
                return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Çıkış yapıldı.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        
        img = image.load_img(file_path, target_size=MODEL_IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = PREPROCESS(img_array)
        
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])

        if probability >= MODEL_THRESHOLD:
            result = 'PNEUMONIA'
            confidence_percent = probability * 100
        else:
            result = 'NORMAL'
            confidence_percent = (1 - probability) * 100
        
        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
        
        img_for_gradcam = image.load_img(file_path, target_size=MODEL_IMG_SIZE)
        img_array_gradcam = image.img_to_array(img_for_gradcam)
        img_array_gradcam = np.expand_dims(img_array_gradcam, axis=0)
        img_array_gradcam = PREPROCESS(img_array_gradcam)
        
        gradcam_success = generate_gradcam(
            file_path,
            img_array_gradcam,
            model,
            heatmap_path,
            layer_name=GRADCAM_LAYER_NAME,
            candidate_layers=GRADCAM_LAYER_CANDIDATES,
            min_intensity=GRADCAM_MIN_INTENSITY,
            apply_threshold=GRADCAM_APPLY_THRESHOLD,
            blur_kernel=GRADCAM_BLUR_KERNEL,
            alpha=GRADCAM_ALPHA,
        )
        
        patient_name = request.form.get('patient_name', '').strip() or 'مجهول'
        patient_age_str = request.form.get('patient_age', None)
        try:
            patient_age = int(patient_age_str) if patient_age_str else None
        except (TypeError, ValueError):
            patient_age = None
        patient_gender = request.form.get('patient_gender', 'غير محدد')
        patient_phone = request.form.get('patient_phone', '').strip()
        tc_kimlik = request.form.get('tc_kimlik', '').strip()
        notes = request.form.get('notes', '').strip()

        if patient_name != 'مجهول':
            is_valid, error_msg = validate_patient_name(patient_name)
            if not is_valid:
                flash(error_msg, 'danger')
                return redirect(url_for('index'))
            
            is_valid, error_msg = validate_tc_kimlik(tc_kimlik)
            if not is_valid:
                flash(error_msg, 'danger')
                return redirect(url_for('index'))
            
            is_valid, formatted_phone = validate_phone(patient_phone)
            if not is_valid:
                flash(formatted_phone, 'danger')
                return redirect(url_for('index'))
            
            patient_phone = formatted_phone if formatted_phone else patient_phone

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
            notes,
            doctor_id=session.get('doctor_id'),
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
@login_required
def dashboard():
    doctor_id = session.get('doctor_id')
    stats = db.get_statistics(doctor_id=doctor_id)
    recent_scans = db.get_all_scans(limit=10, doctor_id=doctor_id)
    recent_patients = db.get_recent_patients(limit=5, doctor_id=doctor_id)
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_scans=recent_scans,
                         recent_patients=recent_patients)

@app.route('/api/scans')
@login_required
def api_scans():
    limit = request.args.get('limit', 50, type=int)
    scans = db.get_all_scans(limit=limit, doctor_id=session.get('doctor_id'))
    return jsonify(scans)

@app.route('/api/statistics')
@login_required
def api_statistics():
    stats = db.get_statistics(doctor_id=session.get('doctor_id'))
    return jsonify(stats)

@app.route('/history')
@login_required
def history():
    search_term = request.args.get('search', '')
    prediction_filter = request.args.get('filter', 'ALL')
    sort_by = request.args.get('sort', 'date')
    sort_order = request.args.get('order', 'desc')
    
    scans = db.search_and_filter_scans(search_term, prediction_filter, sort_by, sort_order, doctor_id=session.get('doctor_id'))
    
    return render_template('history.html', scans=scans, 
                         search_term=search_term, 
                         prediction_filter=prediction_filter,
                         sort_by=sort_by,
                         sort_order=sort_order)

@app.route('/api/scans_timeline')
@login_required
def scans_timeline():
    days = request.args.get('days', 30, type=int)
    data = db.get_scans_by_date_range(days, doctor_id=session.get('doctor_id'))
    return jsonify(data)

@app.route('/delete_scan/<int:scan_id>', methods=['POST'])
@login_required
def delete_scan(scan_id):
    success = db.delete_scan(scan_id, doctor_id=session.get('doctor_id'))
    if success:
        return jsonify({'success': True, 'message': 'Kayıt başarıyla silindi'})
    else:
        return jsonify({'success': False, 'message': 'Kayıt silinemedi'}), 400

@app.route('/generate_report/<int:scan_id>')
@login_required
def generate_report(scan_id):
    scan = db.get_scan_by_id(scan_id, doctor_id=session.get('doctor_id'))

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
        'heatmap_path': os.path.join(app.config['HEATMAP_FOLDER'], scan['heatmap_filename']) if scan.get('heatmap_filename') else None,
        'doctor_name': scan.get('doctor_name') or (g.doctor['name'] if g.doctor else ''),
        'hospital_name': g.doctor['hospital_name'] if g.doctor else None,
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
