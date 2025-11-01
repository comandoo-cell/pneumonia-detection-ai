import sqlite3
from datetime import datetime
import os

DATABASE_PATH = 'pneumonia_detection.db'

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            tc_kimlik TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    try:
        cursor.execute('ALTER TABLE patients ADD COLUMN tc_kimlik TEXT')
    except:
        pass
        
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_filename TEXT NOT NULL,
            heatmap_filename TEXT,
            prediction TEXT NOT NULL,
            confidence REAL,
            notes TEXT,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_scans INTEGER DEFAULT 0,
            pneumonia_cases INTEGER DEFAULT 0,
            normal_cases INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('SELECT COUNT(*) FROM statistics')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO statistics (total_scans, pneumonia_cases, normal_cases)
            VALUES (0, 0, 0)
        ''')
    
    conn.commit()
    conn.close()

def add_patient(name, age, gender, phone, tc_kimlik=''):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO patients (name, age, gender, phone, tc_kimlik)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, age, gender, phone, tc_kimlik))
    
    patient_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return patient_id

def add_scan(patient_id, image_filename, heatmap_filename, prediction, confidence, notes=''):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO scans (patient_id, image_filename, heatmap_filename, prediction, confidence, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, image_filename, heatmap_filename, prediction, confidence, notes))
    
    scan_id = cursor.lastrowid
    
    cursor.execute('UPDATE statistics SET total_scans = total_scans + 1, last_updated = ?', 
                   (datetime.now(),))
    
    if prediction == 'PNEUMONIA':
        cursor.execute('UPDATE statistics SET pneumonia_cases = pneumonia_cases + 1')
    else:
        cursor.execute('UPDATE statistics SET normal_cases = normal_cases + 1')
    
    conn.commit()
    conn.close()
    
    return scan_id

def get_all_scans(limit=50):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.*, p.name as patient_name, p.age, p.gender, p.phone, p.tc_kimlik
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        ORDER BY s.scan_date DESC
        LIMIT ?
    ''', (limit,))
    
    scans = cursor.fetchall()
    conn.close()
    
    return [dict(scan) for scan in scans]

def get_patient_scans(patient_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM scans
        WHERE patient_id = ?
        ORDER BY scan_date DESC
    ''', (patient_id,))
    
    scans = cursor.fetchall()
    conn.close()
    
    return [dict(scan) for scan in scans]

def get_statistics():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM statistics ORDER BY id DESC LIMIT 1')
    stats = cursor.fetchone()
    conn.close()
    
    return dict(stats) if stats else None

def get_recent_patients(limit=10):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT p.*, COUNT(s.id) as scan_count
        FROM patients p
        LEFT JOIN scans s ON p.id = s.patient_id
        GROUP BY p.id
        ORDER BY p.created_at DESC
        LIMIT ?
    ''', (limit,))
    
    patients = cursor.fetchall()
    conn.close()
    
    return [dict(patient) for patient in patients]

def search_patients(search_term):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT p.*, COUNT(s.id) as scan_count
        FROM patients p
        LEFT JOIN scans s ON p.id = s.patient_id
        WHERE p.name LIKE ? OR p.phone LIKE ?
        GROUP BY p.id
        ORDER BY p.created_at DESC
    ''', (f'%{search_term}%', f'%{search_term}%'))
    
    patients = cursor.fetchall()
    conn.close()
    
    return [dict(patient) for patient in patients]

def delete_scan(scan_id):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT prediction FROM scans WHERE id = ?', (scan_id,))
    scan = cursor.fetchone()
    
    if not scan:
        conn.close()
        return False
    
    prediction = scan[0]
    
    cursor.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
    
    cursor.execute('UPDATE statistics SET total_scans = total_scans - 1, last_updated = ?', 
                   (datetime.now(),))
    
    if prediction == 'PNEUMONIA':
        cursor.execute('UPDATE statistics SET pneumonia_cases = pneumonia_cases - 1')
    else:
        cursor.execute('UPDATE statistics SET normal_cases = normal_cases - 1')
    
    conn.commit()
    conn.close()
    
    return True

def search_and_filter_scans(search_term='', prediction_filter='', sort_by='date', sort_order='desc'):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = '''
        SELECT s.*, p.name as patient_name, p.age, p.gender, p.tc_kimlik
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.id
        WHERE 1=1
    '''
    
    params = []
    
    if search_term:
        query += ' AND (p.name LIKE ? OR s.notes LIKE ? OR s.scan_date LIKE ? OR p.tc_kimlik LIKE ?)'
        search_pattern = f'%{search_term}%'
        params.extend([search_pattern, search_pattern, search_pattern, search_pattern])
    
    if prediction_filter and prediction_filter != 'ALL':
        query += ' AND s.prediction = ?'
        params.append(prediction_filter)
    
    if sort_by == 'date':
        query += f' ORDER BY s.scan_date {sort_order.upper()}'
    elif sort_by == 'confidence':
        query += f' ORDER BY s.confidence {sort_order.upper()}'
    elif sort_by == 'name':
        query += f' ORDER BY p.name {sort_order.upper()}'
    
    cursor.execute(query, params)
    scans = cursor.fetchall()
    conn.close()
    
    return [dict(scan) for scan in scans]

def get_scans_by_date_range(days=30):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            DATE(scan_date) as date,
            COUNT(*) as total,
            SUM(CASE WHEN prediction = 'PNEUMONIA' THEN 1 ELSE 0 END) as pneumonia,
            SUM(CASE WHEN prediction = 'NORMAL' THEN 1 ELSE 0 END) as normal
        FROM scans
        WHERE scan_date >= date('now', '-' || ? || ' days')
        GROUP BY DATE(scan_date)
        ORDER BY date ASC
    ''', (days,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in results]

def get_statistics_summary():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM statistics ORDER BY id DESC LIMIT 1')
    stats = cursor.fetchone()
    
    cursor.execute('SELECT AVG(confidence) as avg_confidence FROM scans')
    avg_conf = cursor.fetchone()
    
    cursor.execute('''
        SELECT COUNT(*) as recent_scans 
        FROM scans 
        WHERE scan_date >= date('now', '-7 days')
    ''')
    recent = cursor.fetchone()
    
    conn.close()
    
    result = dict(stats) if stats else {}
    result['avg_confidence'] = round(avg_conf['avg_confidence'], 2) if avg_conf['avg_confidence'] else 0
    result['recent_scans'] = recent['recent_scans'] if recent else 0
    
    return result

if __name__ == '__main__':
    init_db()
