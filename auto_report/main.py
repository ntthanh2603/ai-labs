from report_service import generate_security_report_pdf

# Sample security assessment data for Executive/Management Report
report_data = {
    "system_name": "E-Commerce Platform v2.5",
    "assessment_date": "2026-01-10 to 2026-01-15",
    "scope": "Web Application, API Gateway, Database Layer",
    "security_score": 58,  # 0-100
    "risk_level": "High",  # Critical/High/Medium/Low
    
    "executive_summary": """
        <p><strong>Assessment Overview:</strong> A comprehensive security assessment was conducted on the E-Commerce Platform v2.5 over a 5-day period. The assessment covered the web application, API gateway, and database infrastructure.</p>
        
        <p><strong>Key Findings:</strong> The system currently presents a <strong style="color: #f97316;">HIGH RISK</strong> to the organization. We identified <strong>2 Critical</strong> and <strong>3 High-severity</strong> vulnerabilities that require immediate attention.</p>
        
        <p><strong>Recommendation:</strong> 🚨 <em>Immediate action is required to address the 2 critical vulnerabilities within the next 14 days to prevent potential data breaches and financial losses.</em></p>
    """,
    
    "vulnerabilities": [
        {
            "id": "V-001",
            "name": "SQL Injection in Payment Gateway",
            "severity": "Critical",
            "impact": "Complete database compromise, customer data exposure, financial fraud"
        },
        {
            "id": "V-002",
            "name": "Unauthenticated API Endpoints",
            "severity": "Critical",
            "impact": "Unauthorized access to customer PII, order manipulation"
        },
        {
            "id": "V-003",
            "name": "Weak Password Policy",
            "severity": "High",
            "impact": "Account takeover, unauthorized transactions"
        },
        {
            "id": "V-004",
            "name": "Missing Rate Limiting",
            "severity": "High",
            "impact": "DDoS attacks, service disruption, revenue loss"
        },
        {
            "id": "V-005",
            "name": "Outdated SSL/TLS Configuration",
            "severity": "High",
            "impact": "Man-in-the-middle attacks, data interception"
        },
        {
            "id": "V-006",
            "name": "Insufficient Logging",
            "severity": "Medium",
            "impact": "Delayed incident detection, compliance violations"
        },
        {
            "id": "V-007",
            "name": "Missing CSRF Protection",
            "severity": "Medium",
            "impact": "Unauthorized actions on behalf of users"
        },
        {
            "id": "V-008",
            "name": "Verbose Error Messages",
            "severity": "Low",
            "impact": "Information disclosure to attackers"
        },
    ],
    
    "business_impact": {
        "financial": "Estimated potential loss: $500K - $2M from data breach fines (GDPR), customer compensation, and business disruption. Average cost per compromised record: $150.",
        
        "reputation": "Severe damage to brand trust. Customer churn estimated at 25-40% following a public breach. Recovery timeline: 18-24 months.",
        
        "legal": "Non-compliance with PCI-DSS, GDPR, and CCPA. Potential regulatory fines up to 4% of annual revenue. Class-action lawsuit risk.",
        
        "worst_case": "Complete database breach exposing 500K+ customer records including payment information. Business shutdown for 2-3 weeks. Permanent loss of enterprise clients. Regulatory investigation and criminal charges."
    },
    
    "actions_required": [
        {
            "action": "🔴 URGENT: Patch SQL Injection vulnerability in payment gateway",
            "deadline": "2026-01-30 (14 days)",
            "responsible": "CTO + Engineering Lead"
        },
        {
            "action": "🔴 URGENT: Implement authentication on all API endpoints",
            "deadline": "2026-01-30 (14 days)",
            "responsible": "API Team Lead"
        },
        {
            "action": "🟠 Enforce strong password policy (12+ chars, complexity requirements)",
            "deadline": "2026-02-15 (30 days)",
            "responsible": "Security Team"
        },
        {
            "action": "🟠 Deploy rate limiting across all public endpoints",
            "deadline": "2026-02-15 (30 days)",
            "responsible": "DevOps Team"
        },
        {
            "action": "🟠 Upgrade SSL/TLS to TLS 1.3 and disable weak ciphers",
            "deadline": "2026-02-28 (45 days)",
            "responsible": "Infrastructure Team"
        },
        {
            "action": "📋 Schedule monthly security review meetings with executive team",
            "deadline": "Ongoing",
            "responsible": "CISO"
        }
    ]
}

# Generate the security report
html, pdf_path = generate_security_report_pdf(report_data)

print("=" * 60)
print("✅ Security Report Generated Successfully!")
print("=" * 60)
print(f"📄 PDF Location: {pdf_path}")
print(f"📊 Security Score: {report_data['security_score']}/100")
print(f"⚠️  Risk Level: {report_data['risk_level']}")
print(f"🔍 Vulnerabilities Found: {len(report_data['vulnerabilities'])}")
print("=" * 60)

