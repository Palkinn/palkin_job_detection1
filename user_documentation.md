🛡️ USER DOCUMENTATION: SECUREGUARD SYSTEM
### 1. Introduction
SecureGuard is a next-generation security tool designed to protect job seekers from the growing threat of fraudulent recruitment postings. This manual provides a comprehensive guide for both general users and system administrators to effectively use the SecureGuard interface.

### 2. User Manual for Web Interface
The Web Interface is the primary touchpoint for job seekers to verify the legitimacy of job postings instantly.

### 2.1 Navigating the Homepage
The Input Area: Located at the center of the screen, this is a large, high-contrast text area labeled "Paste Job Description Here."

Analysis Button: After pasting the text, click the "Scan for Risks" button.

Note: To ensure high-quality analysis, the system requires a minimum of 15 characters.

### 2.2 Results Display
The system uses a visual signaling system to communicate the AI's findings:

🟢 Real (Green): Indicates the job post follows legitimate professional patterns.

🔴 Fake (Red): Indicates high-risk fraud patterns (e.g., suspicious contact info, unrealistic salary).

Confidence Score: A percentage indicating how certain the AI is about its prediction.

### 2.3 Reporting a Prediction (Manual Flagging)
If you believe the AI has made an error (False Positive or False Negative):

Click the "Report Error/Flag" button that appears next to the result.

Select a reason from the dropdown (e.g., "Scam Contact Information", "Unrealistic Salary").

Submit the feedback. This moves the record to the Admin Audit Trail for manual verification.

### 3. Admin Panel Guide
The Admin Panel is a restricted, high-security command center meant only for authorized personnel.

### 3.1 Accessing the Dashboard
Step 1: Navigate to the admin.html page.

Step 2 (Authentication): If you are not logged in, the JWT Security Layer will automatically redirect you to the Login Page.

Step 3: Enter your Admin Username and Password. Upon success, a cryptographic token is stored in your browser session.

### 3.2 Monitoring Statistics & KPIs
The dashboard features three main Key Performance Indicators:

Total Analyzed: Every job description processed since the system start.

Fake Detected: Cumulative count of high-risk scams blocked by the engine.

Safe Verified: Validated job posts that passed the security check.

Graphical Analysis: Real-time Pie Charts and Bar Graphs provide a visual representation of fraud ratios.

### 3.3 Administrative Actions
### Audit Trails (Flagged Logs)
At the bottom of the dashboard, you will see the "Audit Trails" table. This displays jobs specifically flagged by users for review.

Columns: Timestamp, Reason, Original Prediction, and Snippet.

### Exporting Data
Export History: Downloads prediction_logs.csv containing every scan.

Export Audit: Downloads flagged_jobs.csv containing user reports.

Pro Tip: These files are pre-formatted for perfect alignment in Microsoft Excel.

### Retraining the Engine
Click "Re-calibrate AI" to trigger the backend model update. This increments the MODEL_VERSION and prepares the system for new data patterns.

### 4. FAQ Section (Frequently Asked Questions)
### Q: How accurate is the detection?
A: SecureGuard operates at 85-95% accuracy. Accuracy is highest when full job descriptions (including responsibilities and requirements) are provided.

### Q: Why do I see "401 Unauthorized"?
A: This is a security feature. It means your session token has expired (default: 60 minutes) or you haven't logged in. Return to login.html.

### Q: What should I do if the "Retrain" button seems stuck?
A: Model calibration involves heavy processing. Please wait 5-10 seconds. If the version number doesn't change, check your terminal for backend errors.

### Q: Does the system save my personal data?
A: No. SecureGuard is privacy-focused. We only log the length of the text and the prediction result for audit purposes.