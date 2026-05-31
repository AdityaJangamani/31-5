# AI-Driven Intelligent Employee Remuneration & Attrition Prediction System

A full-stack HR administration platform powered by open-source Python machine learning models (scikit-learn). The system provides prediction hubs for salary recommendation, attrition probability, promotional readiness, employee categorization, and productivity scoring — all served through a premium glassmorphism dark-mode UI.

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, Vanilla CSS3 (Custom Glassmorphism Dark Mode), Vanilla JS |
| **Backend** | PHP 8+ (with isolated session handling for Admins & Employees) |
| **AI/ML Engine** | Python 3.x, Flask, scikit-learn, Pandas, NumPy |
| **Database** | MySQL 8+ (MariaDB compatible) |
| **Server** | XAMPP (Apache + MySQL on localhost) |
| **Email** | Custom PHP SimpleMailer (SMTP via `.env` config) |

---

## ✨ Features

### AI & Predictions
- ✅ **Salary Prediction** — Linear Regression model recommends fair base salary
- ✅ **Attrition Risk Analysis** — Logistic Regression identifies flight-risk employees
- ✅ **Promotion Readiness** — Random Forest Classifier determines promotion eligibility
- ✅ **Employee Categorization** — Decision Tree classifies employees (High Potential / Steady / Underperformer)
- ✅ **Productivity Scoring** — Weighted composite score from performance, projects, and hours
- ✅ **Bonus & Deduction Calculator** — Rule-based prediction for bonuses and statutory deductions (PF, PT, Income Tax)

### HR Administration (Admin/HR Portal)
- ✅ **Live Employee Directory** — Search, filter, and manage the workforce
- ✅ **Daily Attendance Management** — Real-time sign-in/out tracking with anomaly detection
- ✅ **Salary & Payslip Management** — Add salaries, generate and email digital payslips
- ✅ **Performance Evaluation** — Manager ratings, project tracking, and evaluation history
- ✅ **Bulk Import** — CSV-based mass employee onboarding with template download
- ✅ **Announcements Engine** — Publish, prioritize (Low/Medium/High), and delete company-wide announcements
- ✅ **AI Model Training** — Retrain all 4 models from live database data with one command
- ✅ **Password Reset Requests** — View and approve employee password reset requests

### Employee Self-Service Portal
- ✅ **Personal Dashboard** — View attendance history, salary records, performance metrics
- ✅ **Daily Attendance** — Sign in/out of shifts directly from the portal
- ✅ **Digital Payslips** — View salary breakdowns month by month
- ✅ **Job Satisfaction Updates** — Submit satisfaction feedback (used by AI models)
- ✅ **Announcements Feed** — View important company announcements in real time
- ✅ **Forgot Password Flow** — OTP-based email verification for password recovery
- ✅ **Profile Picture Upload** — Upload and manage profile photos

### Security & Architecture
- ✅ **Bcrypt Password Hashing** — All passwords stored using `password_hash()` (no plain text)
- ✅ **Separated Session Cookies** — Admin (`PHPSESSID`) and Employee (`emp_sess`) sessions are fully isolated
- ✅ **CSRF Token Validation** — All destructive form submissions are CSRF-protected
- ✅ **Role-Based Access Control** — Every admin page verifies session before rendering
- ✅ **XSS Protection** — All user-facing output uses `htmlspecialchars()`

---

## 🚀 Installation & Setup

### Prerequisites
- [XAMPP](https://www.apachefriends.org/) (Apache + MySQL)
- [Python 3.10+](https://www.python.org/downloads/)
- A modern web browser

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Database Setup
1. Start **XAMPP** → Enable **Apache** and **MySQL**.
2. Open phpMyAdmin → `http://localhost/phpmyadmin`
3. Import the main schema:
   ```
   database/schema.sql
   ```
   This creates the `hr_ai_system` database with all tables including `announcements`.
4. Import the daily attendance schema:
   ```
   database/attendance_daily_schema.sql
   ```
5. *(Optional)* Populate with 50 sample employees:
   ```
   database/employees_dataset.sql
   database/salary_dataset.sql
   database/performance_dataset.sql
   database/attendance_dataset.sql
   ```

### 3. Python AI Engine
```bash
cd python

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Train models from live database data
python train_from_db.py

# Start the Flask API server
python app.py
```
> **Note:** The Flask server must remain running on `http://localhost:5000` for AI predictions to work.

### 4. Environment Configuration
Create a `.env` file in the project root for email/SMTP settings:
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_SECURE=tls
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_NAME=HR AI System
```
> **Tip:** For Gmail, enable 2FA and generate an [App Password](https://myaccount.google.com/apppasswords) instead of using your real password.

### 5. Run the Application
1. Place the project folder inside XAMPP's `htdocs/` directory.
2. Open your browser and navigate to:

| Portal | URL |
|---|---|
| **Home Page** | `http://localhost/<folder-name>/` |
| **Admin Login** | `http://localhost/<folder-name>/auth/admin_login.php` |
| **Employee Login** | `http://localhost/<folder-name>/auth/employee_login.php` |

### 6. Default Credentials

| Role | Username / ID | Password |
|---|---|---|
| **Admin** | `admin` | `admin123` |
| **Employee** (sample data) | `EMP001` – `EMP050` | `Firstname@123` (e.g., `John@123`) |

---

## 🤖 ML Models Overview

### 1. Salary Prediction
| Property | Value |
|---|---|
| **Algorithm** | Linear Regression |
| **Purpose** | Predict fair base salary recommendation |
| **Features** | Age, YearsAtCompany, BaseSalary, JobSatisfaction, PerformanceRating, ProjectsCompleted, HoursWorkedPerWeek |
| **Output** | Predicted salary (₹) |

### 2. Attrition Risk
| Property | Value |
|---|---|
| **Algorithm** | Logistic Regression |
| **Purpose** | Identify employees likely to leave |
| **Features** | Same 7 features as above |
| **Output** | Attrition probability (0–100%) |
| **Note** | If no employees have left, proxy attrition labels are generated from risk factors (low satisfaction, high hours, low salary) |

### 3. Promotion Readiness
| Property | Value |
|---|---|
| **Algorithm** | Random Forest Classifier (100 trees) |
| **Purpose** | Determine promotion eligibility |
| **Criteria** | ManagerRating ≥ 4 AND ProjectsCompleted ≥ 10 AND YearsAtCompany ≥ 2 |
| **Output** | Promotion probability (0–100%) |

### 4. Employee Categorization
| Property | Value |
|---|---|
| **Algorithm** | Decision Tree (max depth 6) |
| **Purpose** | Classify employees into career tiers |
| **Categories** | `High Potential` · `Steady Performer` · `Underperformer` |

### Training
```bash
# Train all 4 models from your live MySQL database
python python/train_from_db.py
```

Training results are logged to the `training_data_log` table with accuracy metrics stored as JSON in the `notes` column.

---

## 🔌 Flask API Endpoints

The Flask server runs on `http://localhost:5000` and exposes the following REST endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns API status and available endpoints |
| `POST` | `/predict_salary` | Predict recommended salary |
| `POST` | `/predict_attrition` | Predict attrition risk percentage |
| `POST` | `/predict_promotion` | Predict promotion readiness percentage |
| `POST` | `/predict_intelligent` | Classify employee category |
| `POST` | `/predict_bonus` | Calculate predicted bonus amount |
| `POST` | `/predict_deduction` | Calculate statutory deductions (PF, PT, Income Tax) |
| `POST` | `/analyze_productivity` | Compute weighted productivity score |

### Example Request
```bash
POST http://localhost:5000/predict_salary
Content-Type: application/json

{
  "Age": 35,
  "YearsAtCompany": 5,
  "BaseSalary": 60000,
  "JobSatisfaction": 4,
  "PerformanceRating": 4,
  "ProjectsCompleted": 12,
  "HoursWorkedPerWeek": 45
}
```

---

## 📊 Database Schema (ER Diagram)

```
┌─────────────┐       ┌──────────────┐       ┌──────────────┐
│   users     │       │  employees   │       │ announcements│
├─────────────┤       ├──────────────┤       ├──────────────┤
│ id (PK)     │       │ id (PK)      │       │ id (PK)      │
│ username    │       │ employee_id  │       │ title        │
│ password    │       │ first_name   │       │ content      │
│ role        │       │ last_name    │       │ priority     │
└─────────────┘       │ email        │       │ created_at   │
                      │ password     │       │ author       │
                      │ department   │       └──────────────┘
                      │ job_role     │
                      │ date_joined  │    ┌──────────────────┐
                      │ status       │    │training_data_log │
                      │ age          │    ├──────────────────┤
                      │ years_at_co  │    │ id (PK)          │
                      │ job_satisf.  │    │ total_records    │
                      │ has_left     │    │ records_used     │
                      │ left_date    │    │ training_date    │
                      └──────┬───────┘    │ data_version     │
                             │            │ notes            │
              ┌──────────────┼────────────┘
              │              │
     ┌────────▼───┐  ┌───────▼──────┐  ┌──────────────┐
     │ attendance │  │   salary     │  │ performance  │
     ├────────────┤  ├──────────────┤  ├──────────────┤
     │ id (PK)    │  │ id (PK)      │  │ id (PK)      │
     │ emp_id(FK) │  │ emp_id (FK)  │  │ emp_id (FK)  │
     │ month      │  │ base_salary  │  │ prod_score   │
     │ year       │  │ bonus        │  │ mgr_rating   │
     │ total_days │  │ deductions   │  │ projects     │
     │ days_pres. │  │ net_salary   │  │ eval_date    │
     └────────────┘  │ month / year │  │ hours/week   │
                     └──────────────┘  └──────────────┘

Relationships:
  employees ──< attendance    (1:N, ON DELETE CASCADE)
  employees ──< salary        (1:N, ON DELETE CASCADE)
  employees ──< performance   (1:N, ON DELETE CASCADE)
```

**Additional runtime table:** `attendance_daily` — granular daily sign-in/out records that are automatically aggregated into the `attendance` table.

---

## 📁 Project Structure

```text
├── index.php                        # Public landing page (portal hub)
├── .env                             # Environment config (SMTP credentials)
├── .gitignore                       # Git ignore rules
│
├── auth/
│   ├── admin_login.php              # Admin/HR login page
│   ├── employee_login.php           # Employee login page
│   ├── logout.php                   # Admin logout handler
│   └── employee_logout.php          # Employee logout handler
│
├── admin/
│   ├── dashboard.php                # Admin dashboard (stats + announcements CRUD)
│   ├── add_employee.php             # Add new employee form
│   ├── view_employee.php            # Employee profile viewer
│   ├── add_salary.php               # Add salary records
│   ├── salary.php                   # Salary management overview
│   ├── payslip.php                  # View/generate payslips
│   ├── send_payslip_email.php       # Email payslips to employees
│   ├── manage_attendance.php        # Attendance management & anomaly detection
│   ├── manage_performance.php       # Performance evaluation management
│   ├── bulk_import.php              # CSV bulk employee import
│   ├── import_employees.php         # Import processing backend
│   ├── download_template.php        # CSV template download
│   └── password_requests.php        # Admin password reset request viewer
│
├── employee/
│   ├── employee_dashboard.php       # Employee self-service dashboard
│   ├── forgot_password.php          # OTP-based password recovery
│   └── upload_profile_pic.php       # Profile photo upload handler
│
├── prediction/
│   ├── prediction.php               # AI prediction hub (salary, attrition, promotion)
│   ├── attrition.php                # Detailed attrition analysis page
│   ├── productivity.php             # Productivity analysis page
│   └── promotion.php                # Promotion readiness page
│
├── css/
│   └── style.css                    # Global design system (glassmorphism dark mode)
│
├── php/
│   ├── db.php                       # PDO MySQL connection + auto-migration
│   ├── sidebar.php                  # Admin sidebar navigation component
│   ├── predictions.php              # PHP cURL client for Flask API
│   ├── csrf.php                     # CSRF token generation & validation
│   ├── config.php                   # App configuration loader
│   ├── email_config.php             # SMTP settings from .env
│   └── SimpleMailer.php             # Lightweight SMTP email class
│
├── python/
│   ├── app.py                       # Flask REST API server (port 5000)
│   ├── train_from_db.py             # Train models from MySQL database
│   ├── start_flask_server.bat       # Windows batch launcher
│   ├── start_flask_server.ps1       # PowerShell launcher
│   ├── requirements.txt             # Python dependencies
│   └── venv/                        # Python virtual environment
│
├── models/
│   ├── salary_model.pkl             # Trained Linear Regression model
│   ├── attrition_model.pkl          # Trained Logistic Regression model
│   ├── promotion_model.pkl          # Trained Random Forest model
│   └── category_model.pkl           # Trained Decision Tree model
│
├── database/
│   ├── schema.sql                   # Full database schema (all tables + indexes)
│   ├── attendance_daily_schema.sql  # Daily attendance tracking table
│   ├── employees_dataset.sql        # Sample employee data (50 records)
│   ├── salary_dataset.sql           # Sample salary records
│   ├── performance_dataset.sql      # Sample performance evaluations
│   └── attendance_dataset.sql       # Sample attendance records
│
└── uploads/                         # Employee profile pictures
```

---

## 🐛 Troubleshooting

### "AI Insights return 0 or fail"
- The Python Flask server is not running. Open a terminal:
  ```bash
  cd python
  python app.py
  ```
- Keep the terminal open while using the web app.
- Alternatively, use the provided launcher scripts:
  ```bash
  # Windows (batch)
  python\start_flask_server.bat

  # Windows (PowerShell)
  python\start_flask_server.ps1
  ```

### "Cannot connect to database"
- Ensure XAMPP MySQL is running.
- Check credentials in `php/db.php` and `python/train_from_db.py`.
- Default: `root` with no password on `localhost`.

### "Admin gets logged out when Employee logs in"
- This was fixed via separated session cookies (`emp_sess` vs `PHPSESSID`).
- Clear browser cookies and log in again.

### "Future-Dated Attendance Warning"
- The system flags attendance records with future dates.
- Use the **"Delete Future Records"** button in `admin/manage_attendance.php`.

### "Training fails with fewer than 10 records"
- Ensure employees have matching salary AND performance records in the database.
- Add data through the admin portal first, then retrain.

---

## 📝 License
MIT License — Free for educational and commercial use.
