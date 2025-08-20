import mysql.connector
from collections import defaultdict
from datetime import datetime, timedelta
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import smtplib
from email.message import EmailMessage

# === DB CONFIG ===
db = mysql.connector.connect(
    host="localhost",
    user="kimaiuser",
    password="StrongPassword123!",
    database="kimai2"
)
cursor = db.cursor()

# === TIME RANGE: LAST 7 DAYS ===
end = datetime.now()
start = end - timedelta(days=7)
start_str = start.strftime('%Y-%m-%d %H:%M:%S')
end_str = end.strftime('%Y-%m-%d %H:%M:%S')

# === FETCH WEEKLY DURATION ===
cursor.execute("""
SELECT u.username, p.name AS project, a.name AS activity, SUM(t.duration) AS weekly_duration
FROM kimai2_timesheet t
JOIN kimai2_users u ON t.user = u.id
LEFT JOIN kimai2_projects p ON t.project_id = p.id
LEFT JOIN kimai2_activities a ON t.activity_id = a.id
WHERE t.start_time BETWEEN %s AND %s
GROUP BY u.username, p.name, a.name
""", (start_str, end_str))

weekly_data = cursor.fetchall()

# === FETCH TOTAL DURATION ===
cursor.execute("""
SELECT u.username, SUM(t.duration) AS total_duration
FROM kimai2_timesheet t
JOIN kimai2_users u ON t.user = u.id
GROUP BY u.username
""")
total_data = cursor.fetchall()

# === Build total time map for (username, project, activity) ===
total_map = {u: d for u, d in total_data}

# === PDF Generation ===
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

pdf.cell(200, 10, text="Kimai Weekly Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(200, 10, text=f"Week: {start.strftime('%d-%b-%Y')} to {end.strftime('%d-%b-%Y')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

# Table headers
pdf.set_font("Helvetica", "B", size=11)
pdf.cell(40, 10, text="User", border=1)
pdf.cell(40, 10, text="Project", border=1)
pdf.cell(40, 10, text="Activity", border=1)
pdf.cell(35, 10, text="Weekly Time", border=1)
pdf.cell(35, 10, text="Total Time", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Table rows
pdf.set_font("Helvetica", size=11)
for username, project, activity, weekly_sec in weekly_data:
    total_sec = total_map.get(username, 0)

    # Weekly time
    wh = weekly_sec // 3600
    wm = (weekly_sec % 3600) // 60
    weekly_str = f"{wh}h {wm}m"

    # Total time
    th = total_sec // 3600
    tm = (total_sec % 3600) // 60
    total_str = f"{th}h {tm}m"

    pdf.cell(40, 10, text=username or "-", border=1)
    pdf.cell(40, 10, text=project or "-", border=1)
    pdf.cell(40, 10, text=activity or "-", border=1)
    pdf.cell(35, 10, text=weekly_str, border=1)
    pdf.cell(35, 10, text=total_str, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Save PDF
pdf_file = "/tmp/kimai_db_weekly_report.pdf"
pdf.output(pdf_file)

# === EMAIL CONFIG ===
gmail_user = "yelagandulasupraja@gmail.com"
gmail_app_password = "rvfrdedhonyhjdea"
admin_email = "sanjanaproject36@gmail.com"

msg = EmailMessage()
msg['Subject'] = 'Kimai DB Weekly Report'
msg['From'] = gmail_user
msg['To'] = admin_email
msg.set_content('Attached is the weekly report extracted directly from the Kimai database.')

with open(pdf_file, 'rb') as f:
    file_data = f.read()
    msg.add_attachment(file_data, maintype='application', subtype='pdf', filename='kimai_db_report.pdf')

with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
    smtp.starttls()
    smtp.login(gmail_user, gmail_app_password)
    smtp.send_message(msg)

print("✅ Report sent to admin via email.")