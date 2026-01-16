import io
import base64
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# Set seaborn style for professional charts
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'


def generate_risk_distribution_chart(risk_data: Dict[str, int]) -> str:
    """Generate a pie chart showing risk level distribution"""
    buf = io.BytesIO()
    
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
    
    colors = {
        'Critical': '#dc2626',  # Red
        'High': '#f97316',      # Orange
        'Medium': '#eab308',    # Yellow
        'Low': '#22c55e'        # Green
    }
    
    labels = list(risk_data.keys())
    sizes = list(risk_data.values())
    chart_colors = [colors.get(label, '#6b7280') for label in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=chart_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'},
        explode=[0.05] * len(labels)  # Slightly separate slices
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
    
    ax.set_title('Risk Distribution by Severity', fontsize=13, weight='bold', pad=15)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode()


def generate_severity_bar_chart(vulnerabilities: List[Dict]) -> str:
    """Generate a bar chart showing vulnerability count by severity"""
    buf = io.BytesIO()
    
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    
    severity_counts = {}
    for vuln in vulnerabilities:
        severity = vuln.get('severity', 'Unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    severities = ['Critical', 'High', 'Medium', 'Low']
    counts = [severity_counts.get(s, 0) for s in severities]
    colors_map = ['#dc2626', '#f97316', '#eab308', '#22c55e']
    
    bars = ax.bar(severities, counts, color=colors_map, edgecolor='#333', linewidth=1.5, alpha=0.9)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax.set_ylabel('Number of Vulnerabilities', fontsize=11, weight='bold')
    ax.set_xlabel('Severity Level', fontsize=11, weight='bold')
    ax.set_title('Vulnerability Count by Severity', fontsize=13, weight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(counts) + 1 if max(counts) > 0 else 5)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode()


def generate_security_score_gauge(score: int) -> str:
    """Generate a simple and clear security score visualization"""
    buf = io.BytesIO()
    
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
    
    # Determine color based on score
    if score >= 80:
        color = '#22c55e'  # Green
        rating = 'Good'
    elif score >= 60:
        color = '#eab308'  # Yellow
        rating = 'Fair'
    elif score >= 40:
        color = '#f97316'  # Orange
        rating = 'Poor'
    else:
        color = '#dc2626'  # Red
        rating = 'Critical'
    
    # Create horizontal bar
    ax.barh([0], [score], height=0.6, color=color, alpha=0.8, edgecolor='#333', linewidth=2)
    ax.barh([0], [100-score], left=[score], height=0.6, color='#e5e7eb', alpha=0.5)
    
    # Add score text
    ax.text(score/2, 0, f'{score}', ha='center', va='center', 
            fontsize=32, weight='bold', color='white')
    
    # Add rating text
    ax.text(score + (100-score)/2, 0, rating, ha='center', va='center',
            fontsize=14, weight='bold', color='#6b7280')
    
    # Configure plot
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([])
    ax.set_xlabel('Security Score (0-100)', fontsize=11, weight='bold')
    ax.set_title('Overall Security Score', fontsize=13, weight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode()


def generate_security_report_pdf(report_data: Dict) -> tuple:
    """
    Generate a professional security report PDF
    
    Args:
        report_data: Dictionary containing:
            - system_name: str
            - assessment_date: str
            - scope: str
            - security_score: int (0-100)
            - risk_level: str (Critical/High/Medium/Low)
            - executive_summary: str
            - vulnerabilities: List[Dict] with keys: id, name, severity, impact
            - business_impact: Dict with keys: financial, reputation, legal, worst_case
            - actions_required: List[Dict] with keys: action, deadline, responsible
    """
    
    # Generate charts
    risk_distribution = {
        'Critical': sum(1 for v in report_data['vulnerabilities'] if v['severity'] == 'Critical'),
        'High': sum(1 for v in report_data['vulnerabilities'] if v['severity'] == 'High'),
        'Medium': sum(1 for v in report_data['vulnerabilities'] if v['severity'] == 'Medium'),
        'Low': sum(1 for v in report_data['vulnerabilities'] if v['severity'] == 'Low'),
    }
    # Remove zero values
    risk_distribution = {k: v for k, v in risk_distribution.items() if v > 0}
    
    chart_risk_dist = generate_risk_distribution_chart(risk_distribution)
    chart_severity = generate_severity_bar_chart(report_data['vulnerabilities'])
    chart_score = generate_security_score_gauge(report_data['security_score'])
    
    # Load template
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("security_report.html")
    
    # Render HTML
    html = template.render(
        **report_data,
        chart_risk_distribution=chart_risk_dist,
        chart_severity_bar=chart_severity,
        chart_security_score=chart_score,
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"security_report_{timestamp}.pdf"
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    output_path = os.path.join(output_dir, filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate PDF
    HTML(string=html).write_pdf(output_path)
    
    return html, output_path
