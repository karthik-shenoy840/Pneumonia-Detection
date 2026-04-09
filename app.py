from flask import Flask, render_template, request, redirect, url_for, session, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-secret-key'
MODEL_PATH = 'pneumonia_model.h5'
USER_STORE = 'users.json'
UPLOAD_FOLDER = os.path.join('static', 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(USER_STORE):
    with open(USER_STORE, 'w') as f:
        json.dump({}, f)


def load_users():
    with open(USER_STORE, 'r') as f:
        return json.load(f)


def save_users(users):
    with open(USER_STORE, 'w') as f:
        json.dump(users, f, indent=4)


def get_user(email):
    if not email:
        return None
    return load_users().get(email)


def get_medical_recommendations(prediction, confidence_score):
    """Generate medical recommendations based on prediction and confidence."""
    if prediction == 'NORMAL':
        return {
            'primary_action': 'No immediate medical intervention required',
            'specialist': 'Continue regular health check-ups with your primary care physician',
            'follow_up': 'Schedule routine chest X-ray in 6-12 months if symptoms persist',
            'lifestyle': 'Maintain healthy lifestyle with regular exercise and balanced diet'
        }
    else:  # PNEUMONIA
        severity = get_severity_level(confidence_score)
        if severity == 'Mild':
            return {
                'primary_action': 'Consult primary care physician within 24-48 hours',
                'specialist': 'May require antibiotics and rest; follow up with pulmonologist if symptoms worsen',
                'follow_up': 'Repeat chest X-ray in 2-4 weeks to monitor improvement',
                'lifestyle': 'Rest, stay hydrated, avoid smoking, follow prescribed medication regimen'
            }
        elif severity == 'Moderate':
            return {
                'primary_action': 'Seek medical attention within 24 hours',
                'specialist': 'Consult pulmonologist or infectious disease specialist',
                'follow_up': 'Hospital admission may be required; follow-up imaging in 1-2 weeks',
                'lifestyle': 'Complete antibiotic course, monitor temperature & take immediate action if it worsen'
            }
        else:  # Severe
            return {
                'primary_action': 'URGENT: Seek emergency medical care immediately',
                'specialist': 'Emergency room evaluation, possible ICU admission, consult infectious disease specialist',
                'follow_up': 'Hospital monitoring required; follow-up with specialist within 1 week of discharge',
                'lifestyle': 'Strict bed rest, oxygen therapy if needed, close medical supervision required'
            }

def get_severity_level(confidence_score):
    """Categorize severity based on model confidence."""
    if confidence_score < 0.6:
        return 'Mild'
    elif confidence_score < 0.8:
        return 'Moderate'
    else:
        return 'Severe'

def get_risk_assessment(confidence_score, prediction):
    """Calculate risk assessment score."""
    base_score = confidence_score * 100

    if prediction == 'PNEUMONIA':
        # Higher risk for pneumonia detection
        risk_score = min(95, base_score + 10)
    else:
        # Lower risk for normal results, but account for false negatives
        risk_score = max(5, 100 - base_score)

    return {
        'score': round(risk_score, 1),
        'level': 'High' if risk_score >= 70 else 'Medium' if risk_score >= 40 else 'Low',
        'confidence': round(confidence_score * 100, 1)
    }

def add_history_entry(email, entry):
    users = load_users()
    user = users.get(email)
    if not user:
        return
    history = user.get('history', [])
    history.insert(0, entry)  # Insert at beginning to show most recent first
    user['history'] = history[:30]  # Keep only last 30 entries
    users[email] = user
    save_users(users)

model = tf.keras.models.load_model(MODEL_PATH)

MODEL_INFO = {
    'name': 'Pneumonia Detection CNN',
    'description': 'A lightweight convolutional neural network trained to classify chest X-ray scans as Pneumonia or Normal.',
    'architecture': 'Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense → Dense',
    'input_shape': '(150, 150, 3)',
    'output': 'Binary classification',
    'training_epochs': '10 epochs',
    'accuracy': '92% (example value)',
    'loss': '0.24 (example value)'
}

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        logged_in=('user_email' in session),
        user_name=session.get('user_name'),
        message=session.pop('message', None)
    )

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'user_email' not in session:
        session['message'] = 'Please login to access your dashboard.'
        return redirect(url_for('login'))

    user = get_user(session['user_email'])
    history = user.get('history', []) if user else []
    total_scans = len(history)
    pneumonia_count = sum(1 for item in history if item['prediction'] == 'PNEUMONIA')
    normal_count = sum(1 for item in history if item['prediction'] == 'NORMAL')
    recent = history[0] if history else None

    return render_template(
        'dashboard.html',
        logged_in=True,
        user_name=session.get('user_name'),
        total_scans=total_scans,
        pneumonia_count=pneumonia_count,
        normal_count=normal_count,
        recent=recent
    )

@app.route('/history', methods=['GET'])
def history():
    if 'user_email' not in session:
        session['message'] = 'Please login to see your history.'
        return redirect(url_for('login'))

    user = get_user(session['user_email'])
    history = user.get('history', []) if user else []
    return render_template(
        'history.html',
        logged_in=True,
        user_name=session.get('user_name'),
        history=history
    )

@app.route('/model', methods=['GET'])
def model_info():
    if 'user_email' not in session:
        session['message'] = 'Please login to view model details.'
        return redirect(url_for('login'))

    return render_template(
        'model.html',
        logged_in=True,
        user_name=session.get('user_name'),
        model_info=MODEL_INFO
    )

@app.route('/results', methods=['GET'])
def results():
    if 'user_email' not in session:
        session['message'] = 'Please login to view the latest result.'
        return redirect(url_for('login'))

    user = get_user(session['user_email'])
    history = user.get('history', []) if user else []
    latest = history[0] if history else None

    return render_template(
        'results.html',
        logged_in=True,
        user_name=session.get('user_name'),
        latest=latest
    )

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_email' not in session:
        session['message'] = 'Please login to view your profile.'
        return redirect(url_for('login'))

    email = session.get('user_email')
    message = None
    error = None

    if request.method == 'POST':
        action = request.form.get('action')
        users = load_users()
        user = users.get(email)

        if action == 'update_info':
            name = request.form.get('name', '').strip()
            if name:
                user['name'] = name
                session['user_name'] = name
                save_users(users)
                message = 'Name updated successfully.'
            else:
                error = 'Name cannot be empty.'

        elif action == 'change_password':
            old_password = request.form.get('old_password', '').strip()
            new_password = request.form.get('new_password', '').strip()
            confirm_password = request.form.get('confirm_password', '').strip()

            if not check_password_hash(user['password_hash'], old_password):
                error = 'Current password is incorrect.'
            elif new_password != confirm_password:
                error = 'New passwords do not match.'
            elif len(new_password) < 6:
                error = 'Password must be at least 6 characters.'
            else:
                user['password_hash'] = generate_password_hash(new_password)
                save_users(users)
                message = 'Password changed successfully.'

    user = get_user(email)
    created_date = user.get('created_date', 'Unknown') if user else 'Unknown'

    return render_template(
        'profile.html',
        logged_in=True,
        user_name=session.get('user_name'),
        user_email=email,
        created_date=created_date,
        message=message,
        error=error
    )

@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    if 'user_email' not in session:
        session['message'] = 'Please login to download results.'
        return redirect(url_for('login'))

    user = get_user(session['user_email'])
    history = user.get('history', []) if user else []
    latest = history[0] if history else None

    if not latest:
        session['message'] = 'No results to download.'
        return redirect(url_for('results'))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,  # Reduced from 24 to 20
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=20,  # Reduced from 30 to 20
        alignment=1
    )
    elements.append(Paragraph('Comprehensive Pneumonia Diagnostic Report', title_style))
    elements.append(Spacer(1, 0.2*inch))  # Reduced from 0.3 to 0.2

    # Patient Information
    patient_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,  # Reduced from 16 to 14
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=12  # Reduced from 15 to 12
    )
    elements.append(Paragraph('Patient Information', patient_title))

    patient_data = [
        ['Patient Name', session.get('user_name')],
        ['Email Address', session.get('user_email')],
        ['Report Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Scan Date', latest['date']],
        ['Image File', latest['image']],
    ]

    patient_table = Table(patient_data, colWidths=[1.8*inch, 4.2*inch])  # Adjusted to match other tables
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Reduced from 12 to 10
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # Reduced from 8 to 6
        ('TOPPADDING', (0, 0), (-1, -1), 6),    # Reduced from 8 to 6
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.2*inch))  # Reduced from 0.3 to 0.2

    # Diagnostic Results
    elements.append(Paragraph('Diagnostic Results', patient_title))

    diagnostic_data = [
        ['Prediction', latest['prediction']],
        ['Confidence Level', f"{latest.get('confidence_score', 0) * 100:.1f}%"],
        ['Severity Level', latest.get('severity', 'Unknown')],
        ['Risk Assessment', f"{latest.get('risk_assessment', {}).get('score', 0):.1f}% ({latest.get('risk_assessment', {}).get('level', 'Unknown')} Risk)"],
    ]

    diagnostic_table = Table(diagnostic_data, colWidths=[1.8*inch, 4.2*inch])  # Adjusted to match other tables
    diagnostic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),  # Reduced from 12 to 10
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # Reduced from 8 to 6
        ('TOPPADDING', (0, 0), (-1, -1), 6),    # Reduced from 8 to 6
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0ea5e9'))
    ]))
    elements.append(diagnostic_table)
    elements.append(Spacer(1, 0.2*inch))  # Reduced from 0.3 to 0.2

    # Medical Recommendations
    elements.append(Paragraph('Medical Recommendations', patient_title))

    recommendations = latest.get('recommendations', {})
    rec_data = [
        ['Primary Action Required', recommendations.get('primary_action', 'N/A')],
        ['Specialist Consultation', recommendations.get('specialist', 'N/A')],
        ['Follow-up Care', recommendations.get('follow_up', 'N/A')],
        ['Lifestyle & Self-Care', recommendations.get('lifestyle', 'N/A')],
    ]

    # Adjust column widths to fit within page borders (total ~6.5 inches for letter size with margins)
    rec_table = Table(rec_data, colWidths=[1.8*inch, 4.2*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0fdf4')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),  # Reduced from 10 to 9
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),  # Reduced from 4 to 3
        ('TOPPADDING', (0, 0), (-1, -1), 3),    # Reduced from 4 to 3
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#16a34a')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('WORDWRAP', (1, 0), (1, -1), True),  # Enable word wrapping for content column
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 0.2*inch))  # Reduced from 0.3 to 0.2

    # Important Notes
    elements.append(Paragraph('Important Medical Notes', patient_title))

    notes_style = ParagraphStyle(
        'NotesStyle',
        parent=styles['Normal'],
        fontSize=9,  # Reduced from 11 to 9
        textColor=colors.HexColor('#dc2626'),
        spaceAfter=8  # Reduced from 10 to 8
    )

    elements.append(Paragraph(
        '⚠️ <b>CRITICAL:</b> This is an AI-generated preliminary assessment and should NOT replace professional medical diagnosis. '
        'Please consult with a qualified healthcare provider for proper evaluation and treatment.',
        notes_style
    ))

    elements.append(Paragraph(
        '📋 <b>Report Interpretation:</b> This analysis is based on chest X-ray imaging using artificial intelligence. '
        'False positives and false negatives can occur. Clinical correlation with symptoms, physical examination, '
        'and additional tests is essential.',
        ParagraphStyle('NormalSmall', parent=styles['Normal'], fontSize=9, spaceAfter=6)  # Smaller font
    ))

    elements.append(Paragraph(
        '🏥 <b>Emergency Situations:</b> If you experience severe symptoms such as difficulty breathing, '
        'chest pain, high fever, or confusion, seek immediate emergency medical care.',
        ParagraphStyle('NormalSmall', parent=styles['Normal'], fontSize=9, spaceAfter=6)  # Smaller font
    ))

    elements.append(Spacer(1, 0.15*inch))  # Reduced spacing

    # Footer
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=8,  # Reduced from 9 to 8
        textColor=colors.HexColor('#64748b'),
        alignment=1
    )

    elements.append(Paragraph(
        f'Report generated by AI Pneumonia Diagnostic Tool on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
        'This report is confidential and intended for the patient only.',
        footer_style
    ))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'pneumonia_medical_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = request.args.get('message')
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        user = get_user(email)

        if user and check_password_hash(user['password_hash'], password):
            session['user_email'] = email
            session['user_name'] = user['name']
            return redirect(url_for('dashboard'))

        message = 'Invalid email or password. Please try again.'

    return render_template('login.html', message=message)

@app.route('/register', methods=['GET', 'POST'])
def register():
    message = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        users = load_users()

        if not name or not email or not password or not confirm_password:
            message = 'Please fill in all fields.'
        elif password != confirm_password:
            message = 'Passwords do not match.'
        elif email in users:
            message = 'An account already exists with this email.'
        else:
            users[email] = {
                'name': name,
                'password_hash': generate_password_hash(password),
                'history': [],
                'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_users(users)
            return redirect(url_for('login', message='Account created successfully! Please login.'))

    return render_template('register.html', message=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_email' not in session:
        session['message'] = 'Please login to upload and analyze an image.'
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return redirect(url_for('dashboard'))

    file = request.files['file']
    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    img = image.load_img(path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    confidence_score = float(result[0][0])
    prediction = 'PNEUMONIA' if confidence_score > 0.5 else 'NORMAL'

    # Calculate medical features
    severity = get_severity_level(confidence_score)
    risk_assessment = get_risk_assessment(confidence_score, prediction)
    recommendations = get_medical_recommendations(prediction, confidence_score)

    entry = {
        'image': filename,
        'prediction': prediction,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'confidence_score': confidence_score,
        'severity': severity,
        'risk_assessment': risk_assessment,
        'recommendations': recommendations
    }
    add_history_entry(session['user_email'], entry)
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)
