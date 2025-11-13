import sqlite3
from datetime import datetime

DATABASE_PATH = 'pneumonia_detection.db'


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def column_exists(conn, table, column):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            tc_kimlik TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            hospital_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_filename TEXT NOT NULL,
            heatmap_filename TEXT,
            prediction TEXT NOT NULL,
            confidence REAL,
            notes TEXT,
            doctor_id INTEGER,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(id)
        )
        '''
    )

    if not column_exists(conn, 'scans', 'doctor_id'):
        cursor.execute('ALTER TABLE scans ADD COLUMN doctor_id INTEGER REFERENCES doctors(id)')

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_scans INTEGER DEFAULT 0,
            pneumonia_cases INTEGER DEFAULT 0,
            normal_cases INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    cursor.execute('SELECT COUNT(*) FROM statistics')
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            '''
            INSERT INTO statistics (total_scans, pneumonia_cases, normal_cases)
            VALUES (0, 0, 0)
            '''
        )

    conn.commit()
    conn.close()


def add_patient(name, age, gender, phone, tc_kimlik=''):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO patients (name, age, gender, phone, tc_kimlik)
        VALUES (?, ?, ?, ?, ?)
        ''',
        (name, age, gender, phone, tc_kimlik),
    )
    patient_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return patient_id


def add_scan(patient_id, image_filename, heatmap_filename, prediction, confidence, notes='', doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO scans (patient_id, image_filename, heatmap_filename, prediction, confidence, notes, doctor_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (patient_id, image_filename, heatmap_filename, prediction, confidence, notes, doctor_id),
    )
    scan_id = cursor.lastrowid

    cursor.execute(
        'UPDATE statistics SET total_scans = total_scans + 1, last_updated = ?',
        (datetime.now(),),
    )
    if prediction == 'PNEUMONIA':
        cursor.execute('UPDATE statistics SET pneumonia_cases = pneumonia_cases + 1')
    else:
        cursor.execute('UPDATE statistics SET normal_cases = normal_cases + 1')

    conn.commit()
    conn.close()
    return scan_id


def get_all_scans(limit=50, doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = (
        'SELECT s.*, p.name as patient_name, p.age, p.gender, p.phone, p.tc_kimlik, '
        'd.name as doctor_name, d.hospital_name '
        'FROM scans s '
        'LEFT JOIN patients p ON s.patient_id = p.id '
        'LEFT JOIN doctors d ON s.doctor_id = d.id '
    )
    params = []
    if doctor_id:
        query += 'WHERE s.doctor_id = ? '
        params.append(doctor_id)
    query += 'ORDER BY s.scan_date DESC LIMIT ?'
    params.append(limit)
    cursor.execute(query, params)
    scans = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return scans


def get_patient_scans(patient_id, doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = 'SELECT * FROM scans WHERE patient_id = ?'
    params = [patient_id]
    if doctor_id:
        query += ' AND doctor_id = ?'
        params.append(doctor_id)
    query += ' ORDER BY scan_date DESC'
    cursor.execute(query, params)
    scans = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return scans


def get_statistics(doctor_id=None):
    if doctor_id is None:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM statistics ORDER BY id DESC LIMIT 1')
        stats = cursor.fetchone()
        conn.close()
        return dict(stats) if stats else None

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT
            COUNT(*) as total_scans,
            SUM(CASE WHEN prediction = 'PNEUMONIA' THEN 1 ELSE 0 END) as pneumonia_cases,
            SUM(CASE WHEN prediction = 'NORMAL' THEN 1 ELSE 0 END) as normal_cases
        FROM scans
        WHERE doctor_id = ?
        ''',
        (doctor_id,),
    )
    stats = cursor.fetchone()
    conn.close()
    if stats:
        return {
            'total_scans': stats['total_scans'] or 0,
            'pneumonia_cases': stats['pneumonia_cases'] or 0,
            'normal_cases': stats['normal_cases'] or 0,
        }
    return {'total_scans': 0, 'pneumonia_cases': 0, 'normal_cases': 0}


def get_recent_patients(limit=10, doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = (
        'SELECT p.*, COUNT(s.id) as scan_count '
        'FROM patients p '
        'LEFT JOIN scans s ON p.id = s.patient_id '
    )
    params = []
    if doctor_id:
        query += 'WHERE s.doctor_id = ? '
        params.append(doctor_id)
    query += 'GROUP BY p.id ORDER BY p.created_at DESC LIMIT ?'
    params.append(limit)
    cursor.execute(query, params)
    patients = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return patients


def search_patients(search_term):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT p.*, COUNT(s.id) as scan_count
        FROM patients p
        LEFT JOIN scans s ON p.id = s.patient_id
        WHERE p.name LIKE ? OR p.phone LIKE ?
        GROUP BY p.id
        ORDER BY p.created_at DESC
        ''',
        (f'%{search_term}%', f'%{search_term}%'),
    )
    patients = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return patients


def delete_scan(scan_id, doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    if doctor_id:
        cursor.execute('SELECT prediction FROM scans WHERE id = ? AND doctor_id = ?', (scan_id, doctor_id))
    else:
        cursor.execute('SELECT prediction FROM scans WHERE id = ?', (scan_id,))
    scan = cursor.fetchone()
    if not scan:
        conn.close()
        return False
    prediction = scan['prediction'] if isinstance(scan, sqlite3.Row) else scan[0]
    if doctor_id:
        cursor.execute('DELETE FROM scans WHERE id = ? AND doctor_id = ?', (scan_id, doctor_id))
    else:
        cursor.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
    cursor.execute(
        'UPDATE statistics SET total_scans = total_scans - 1, last_updated = ?',
        (datetime.now(),),
    )
    if prediction == 'PNEUMONIA':
        cursor.execute('UPDATE statistics SET pneumonia_cases = pneumonia_cases - 1')
    else:
        cursor.execute('UPDATE statistics SET normal_cases = normal_cases - 1')
    conn.commit()
    conn.close()
    return True


def search_and_filter_scans(search_term='', prediction_filter='', sort_by='date', sort_order='desc', doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = (
        'SELECT s.*, p.name as patient_name, p.age, p.gender, p.tc_kimlik, '
        'd.name as doctor_name '
        'FROM scans s '
        'LEFT JOIN patients p ON s.patient_id = p.id '
        'LEFT JOIN doctors d ON s.doctor_id = d.id '
        'WHERE 1=1 '
    )
    params = []
    if doctor_id:
        query += 'AND s.doctor_id = ? '
        params.append(doctor_id)
    if search_term:
        query += 'AND (p.name LIKE ? OR s.notes LIKE ? OR s.scan_date LIKE ? OR p.tc_kimlik LIKE ?) '
        search_pattern = f'%{search_term}%'
        params.extend([search_pattern, search_pattern, search_pattern, search_pattern])
    if prediction_filter and prediction_filter != 'ALL':
        query += 'AND s.prediction = ? '
        params.append(prediction_filter)
    if sort_by == 'date':
        query += f'ORDER BY s.scan_date {sort_order.upper()}'
    elif sort_by == 'confidence':
        query += f'ORDER BY s.confidence {sort_order.upper()}'
    elif sort_by == 'name':
        query += f'ORDER BY p.name {sort_order.upper()}'
    cursor.execute(query, params)
    scans = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return scans


def get_scans_by_date_range(days=30, doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = (
        'SELECT DATE(scan_date) as date, '
        'COUNT(*) as total, '
        "SUM(CASE WHEN prediction = 'PNEUMONIA' THEN 1 ELSE 0 END) as pneumonia, "
        "SUM(CASE WHEN prediction = 'NORMAL' THEN 1 ELSE 0 END) as normal "
        'FROM scans '
        "WHERE scan_date >= date('now', '-' || ? || ' days') "
    )
    params = [days]
    if doctor_id:
        query += 'AND doctor_id = ? '
        params.append(doctor_id)
    query += 'GROUP BY DATE(scan_date) ORDER BY date ASC'
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_statistics_summary(doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    if doctor_id:
        cursor.execute(
            '''
            SELECT 
                COUNT(*) as total_scans,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN scan_date >= date('now', '-7 days') THEN 1 ELSE 0 END) as recent_scans
            FROM scans
            WHERE doctor_id = ?
            ''',
            (doctor_id,),
        )
        stats = cursor.fetchone()
        conn.close()
        if not stats:
            return {'total_scans': 0, 'avg_confidence': 0, 'recent_scans': 0}
        return {
            'total_scans': stats['total_scans'] or 0,
            'avg_confidence': round(stats['avg_confidence'], 2) if stats['avg_confidence'] else 0,
            'recent_scans': stats['recent_scans'] or 0,
        }

    cursor.execute('SELECT * FROM statistics ORDER BY id DESC LIMIT 1')
    stats = cursor.fetchone()
    cursor.execute('SELECT AVG(confidence) as avg_confidence FROM scans')
    avg_conf = cursor.fetchone()
    cursor.execute(
        '''
        SELECT COUNT(*) as recent_scans 
        FROM scans 
        WHERE scan_date >= date('now', '-7 days')
        '''
    )
    recent = cursor.fetchone()
    conn.close()
    result = dict(stats) if stats else {}
    result['avg_confidence'] = round(avg_conf['avg_confidence'], 2) if avg_conf['avg_confidence'] else 0
    result['recent_scans'] = recent['recent_scans'] if recent else 0
    return result


def add_doctor(name, username, password_hash, hospital_name=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO doctors (name, username, password_hash, hospital_name)
        VALUES (?, ?, ?, ?)
        ''',
        (name, username, password_hash, hospital_name),
    )
    doctor_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doctor_id


def get_doctor_by_username(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM doctors WHERE username = ?', (username,))
    doctor = cursor.fetchone()
    conn.close()
    return dict(doctor) if doctor else None


def get_doctor_by_id(doctor_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM doctors WHERE id = ?', (doctor_id,))
    doctor = cursor.fetchone()
    conn.close()
    return dict(doctor) if doctor else None


def get_scan_by_id(scan_id, doctor_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    query = (
        'SELECT s.*, p.name as patient_name, p.age, p.gender, p.phone, p.tc_kimlik, '
        'd.name as doctor_name, d.hospital_name '
        'FROM scans s '
        'LEFT JOIN patients p ON s.patient_id = p.id '
        'LEFT JOIN doctors d ON s.doctor_id = d.id '
        'WHERE s.id = ? '
    )
    params = [scan_id]
    if doctor_id:
        query += 'AND s.doctor_id = ? '
        params.append(doctor_id)
    cursor.execute(query, params)
    scan = cursor.fetchone()
    conn.close()
    return dict(scan) if scan else None


if __name__ == '__main__':
    init_db()
