# AI Pneumonia Detection System

A comprehensive web application for pneumonia detection using artificial intelligence and chest X-ray analysis.

## Features

- **AI-Powered Detection**: Uses TensorFlow CNN model for pneumonia classification
- **User Authentication**: Secure login and registration system
- **Medical Reports**: Detailed PDF reports with medical recommendations
- **Dashboard**: User statistics and scan history
- **Risk Assessment**: Severity categorization and risk scoring
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **Backend**: Flask (Python)
- **AI/ML**: TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Database**: JSON file storage
- **PDF Generation**: ReportLab

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Pneumonia_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the trained model file `pneumonia_model.h5` and place it in the root directory

4. Run the application:
```bash
python app.py
```

5. Open your browser and visit `http://localhost:5000`

## Usage

1. Register a new account or login
2. Upload a chest X-ray image
3. View AI analysis results with confidence scores
4. Download detailed medical reports
5. Access scan history and dashboard

## Project Structure

```
Pneumonia_Detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── pneumonia_model.h5     # Trained ML model (not in repo)
├── users.json            # User data (not in repo)
├── static/
│   ├── css/
│   │   └── style.css     # Application styles
│   └── uploads/          # Uploaded images (not in repo)
├── templates/            # HTML templates
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── history.html
│   ├── results.html
│   ├── profile.html
│   └── model.html
├── dataset/              # Training data
└── __pycache__/         # Python cache (ignored)
```

## Medical Disclaimer

⚠️ **IMPORTANT**: This application is for educational and research purposes only. The AI predictions should NOT replace professional medical diagnosis. Always consult with qualified healthcare providers for proper medical evaluation and treatment.

## License

This project is for educational purposes. Please ensure compliance with medical data privacy regulations (HIPAA, GDPR, etc.) when deploying in real environments.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Security Note

- User passwords are hashed using Werkzeug security
- Never commit sensitive data or model files to version control
- Use environment variables for configuration in production