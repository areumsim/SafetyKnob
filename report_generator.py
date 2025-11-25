"""
report_generator.py
분석 결과를 HTML, PDF, MD 보고서로 생성하는 모듈
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# HTML/PDF 생성용
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ReportGenerator:
    """분석 결과 보고서 생성기"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
    def generate_complete_report(self, analysis_results: Dict, report_type: str = "all"):
        """완전한 분석 보고서 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 보고서 데이터 준비
        report_data = self._prepare_report_data(analysis_results)
        
        # 시각화 생성
        charts = self._generate_visualizations(report_data)
        
        if report_type in ["html", "all"]:
            html_file = self.output_dir / f"analysis_report_{timestamp}.html"
            self._generate_html_report(report_data, charts, html_file)
            
        if report_type in ["pdf", "all"] and WEASYPRINT_AVAILABLE:
            pdf_file = self.output_dir / f"analysis_report_{timestamp}.pdf"
            self._generate_pdf_report(report_data, charts, pdf_file)
            
        if report_type in ["md", "all"]:
            md_file = self.output_dir / f"analysis_report_{timestamp}.md"
            self._generate_markdown_report(report_data, charts, md_file)
            
        return {
            "html": html_file if report_type in ["html", "all"] else None,
            "pdf": pdf_file if report_type in ["pdf", "all"] and WEASYPRINT_AVAILABLE else None,
            "md": md_file if report_type in ["md", "all"] else None,
            "timestamp": timestamp
        }
    
    def _prepare_report_data(self, analysis_results: Dict) -> Dict:
        """보고서 데이터 준비"""
        return {
            "title": "SafetyKnob 이미지 안전성 분류 분석 보고서",
            "timestamp": datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분"),
            "analysis_type": analysis_results.get("analysis_type", "Unknown"),
            "model_info": analysis_results.get("model_info", {}),
            "performance": analysis_results.get("performance", {}),
            "validation": analysis_results.get("validation", {}),
            "predictions": analysis_results.get("predictions", {}),
            "statistics": analysis_results.get("statistics", {}),
            "images": analysis_results.get("image_paths", [])
        }
    
    def _generate_visualizations(self, report_data: Dict) -> Dict:
        """시각화 차트 생성"""
        charts = {}
        
        # 1. 성능 메트릭 차트
        if "performance" in report_data and report_data["performance"]:
            charts["performance"] = self._create_performance_chart(report_data["performance"])
        
        # 2. 모델 비교 차트
        if "model_info" in report_data and report_data["model_info"]:
            charts["model_comparison"] = self._create_model_comparison_chart(report_data["model_info"])
        
        # 3. 예측 분포 차트
        if "predictions" in report_data and report_data["predictions"]:
            charts["prediction_distribution"] = self._create_prediction_distribution_chart(report_data["predictions"])
        
        # 4. 신뢰도 분포 차트
        if "statistics" in report_data and report_data["statistics"]:
            charts["confidence_distribution"] = self._create_confidence_chart(report_data["statistics"])
        
        return charts
    
    def _create_performance_chart(self, performance_data: Dict) -> str:
        """성능 메트릭 차트 생성"""
        if PLOTLY_AVAILABLE:
            return self._create_plotly_performance_chart(performance_data)
        else:
            return self._create_matplotlib_performance_chart(performance_data)
    
    def _create_plotly_performance_chart(self, performance_data: Dict) -> str:
        """Plotly로 성능 차트 생성"""
        fig = go.Figure()
        
        # 모델별 성능 데이터 추출
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model, metrics in performance_data.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                models.append(model)
                accuracies.append(metrics.get("accuracy", 0))
                precisions.append(metrics.get("precision", 0))
                recalls.append(metrics.get("recall", 0))
                f1_scores.append(metrics.get("f1_score", 0))
        
        if models:
            fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracies))
            fig.add_trace(go.Bar(name='Precision', x=models, y=precisions))
            fig.add_trace(go.Bar(name='Recall', x=models, y=recalls))
            fig.add_trace(go.Bar(name='F1-Score', x=models, y=f1_scores))
            
            fig.update_layout(
                title='모델 성능 비교',
                barmode='group',
                yaxis_title='점수',
                xaxis_title='모델'
            )
        
        chart_path = self.images_dir / "performance_chart.html"
        fig.write_html(str(chart_path))
        return str(chart_path)
    
    def _create_matplotlib_performance_chart(self, performance_data: Dict) -> str:
        """Matplotlib으로 성능 차트 생성"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 데이터 준비
        models = []
        metrics_data = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
        
        for model, metrics in performance_data.items():
            if isinstance(metrics, dict):
                models.append(model)
                for metric in metrics_data.keys():
                    metrics_data[metric].append(metrics.get(metric, 0))
        
        if models:
            x = np.arange(len(models))
            width = 0.2
            
            for i, (metric, values) in enumerate(metrics_data.items()):
                ax.bar(x + i*width, values, width, label=metric.capitalize())
            
            ax.set_xlabel('모델')
            ax.set_ylabel('점수')
            ax.set_title('모델 성능 비교')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.set_ylim(0, 1)
        
        chart_path = self.images_dir / "performance_chart.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_prediction_distribution_chart(self, predictions_data: Dict) -> str:
        """예측 분포 차트 생성"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 예측 결과 집계
        prediction_counts = {}
        if "summary" in predictions_data:
            prediction_counts = predictions_data["summary"]
        else:
            # predictions에서 직접 계산
            predictions = []
            for pred_data in predictions_data.values():
                if isinstance(pred_data, dict) and "prediction" in pred_data:
                    predictions.append(pred_data["prediction"])
            
            from collections import Counter
            prediction_counts = dict(Counter(predictions))
        
        if prediction_counts:
            labels = list(prediction_counts.keys())
            sizes = list(prediction_counts.values())
            colors = ['lightgreen' if label == 'safe' else 'lightcoral' if label == 'danger' else 'lightgray' for label in labels]
            
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('예측 결과 분포')
        
        chart_path = self.images_dir / "prediction_distribution.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_confidence_chart(self, statistics_data: Dict) -> str:
        """신뢰도 분포 차트 생성"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 신뢰도 데이터 추출
        confidences = []
        if "confidences" in statistics_data:
            confidences = statistics_data["confidences"]
        elif "average_confidence" in statistics_data:
            # 단일 값인 경우 히스토그램 대신 바 차트
            ax.bar(['평균 신뢰도'], [statistics_data["average_confidence"]])
            ax.set_ylabel('신뢰도')
            ax.set_title('평균 신뢰도')
            ax.set_ylim(0, 1)
        
        if confidences:
            ax.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('신뢰도')
            ax.set_ylabel('빈도')
            ax.set_title('신뢰도 분포')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'평균: {np.mean(confidences):.3f}')
            ax.legend()
        
        chart_path = self.images_dir / "confidence_distribution.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_model_comparison_chart(self, model_info: Dict) -> str:
        """모델 비교 차트 생성"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if "training_stats" in model_info:
            models = []
            accuracies = []
            
            for model, stats in model_info["training_stats"].items():
                models.append(model)
                accuracies.append(stats.get("accuracy", 0))
            
            if models:
                ax.bar(models, accuracies, alpha=0.7)
                ax.set_xlabel('모델')
                ax.set_ylabel('정확도')
                ax.set_title('모델별 훈련 정확도')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
        
        chart_path = self.images_dir / "model_comparison.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _generate_html_report(self, report_data: Dict, charts: Dict, output_file: Path):
        """HTML 보고서 생성"""
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_data['title']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .status-safe {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-danger {{
            color: #dc3545;
            font-weight: bold;
        }}
        .status-unknown {{
            color: #6c757d;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{report_data['title']}</h1>
            <p>생성 시간: {report_data['timestamp']}</p>
            <p>분석 유형: {report_data['analysis_type']}</p>
        </div>

        {self._generate_summary_section_html(report_data)}
        {self._generate_performance_section_html(report_data, charts)}
        {self._generate_model_info_section_html(report_data)}
        {self._generate_predictions_section_html(report_data)}
        {self._generate_charts_section_html(charts)}

        <div class="footer">
            <p>🔍 SafetyKnob 이미지 안전성 분류 시스템</p>
            <p>보고서 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_summary_section_html(self, report_data: Dict) -> str:
        """요약 섹션 HTML 생성"""
        summary_stats = self._extract_summary_stats(report_data)
        
        metrics_html = ""
        for label, value, unit in summary_stats:
            metrics_html += f"""
            <div class="metric-card">
                <div class="metric-value">{value}{unit}</div>
                <div class="metric-label">{label}</div>
            </div>"""
        
        return f"""
        <div class="section">
            <h2>📊 분석 요약</h2>
            <div class="metrics-grid">
                {metrics_html}
            </div>
        </div>"""
    
    def _generate_performance_section_html(self, report_data: Dict, charts: Dict) -> str:
        """성능 섹션 HTML 생성"""
        if "performance" not in report_data or not report_data["performance"]:
            return ""
        
        performance_table = self._create_performance_table_html(report_data["performance"])
        chart_html = ""
        
        if "performance" in charts:
            chart_path = charts["performance"]
            if chart_path.endswith('.html'):
                with open(chart_path, 'r', encoding='utf-8') as f:
                    chart_html = f'<div class="chart-container">{f.read()}</div>'
            else:
                chart_html = f'<div class="chart-container"><img src="{os.path.relpath(chart_path, self.output_dir)}" alt="성능 차트"></div>'
        
        return f"""
        <div class="section">
            <h2>🎯 모델 성능</h2>
            {performance_table}
            {chart_html}
        </div>"""
    
    def _generate_model_info_section_html(self, report_data: Dict) -> str:
        """모델 정보 섹션 HTML 생성"""
        if "model_info" not in report_data or not report_data["model_info"]:
            return ""
        
        model_info = report_data["model_info"]
        info_html = "<ul>"
        
        if "model_types" in model_info:
            info_html += f"<li><strong>사용된 모델:</strong> {', '.join(model_info['model_types'])}</li>"
        
        if "is_trained" in model_info:
            status = "✅ 훈련됨" if model_info["is_trained"] else "❌ 미훈련"
            info_html += f"<li><strong>훈련 상태:</strong> {status}</li>"
        
        info_html += "</ul>"
        
        return f"""
        <div class="section">
            <h2>🤖 모델 정보</h2>
            {info_html}
        </div>"""
    
    def _generate_predictions_section_html(self, report_data: Dict) -> str:
        """예측 결과 섹션 HTML 생성"""
        if "predictions" not in report_data or not report_data["predictions"]:
            return ""
        
        predictions_table = self._create_predictions_table_html(report_data["predictions"])
        
        return f"""
        <div class="section">
            <h2>🔍 예측 결과</h2>
            {predictions_table}
        </div>"""
    
    def _generate_charts_section_html(self, charts: Dict) -> str:
        """차트 섹션 HTML 생성"""
        if not charts:
            return ""
        
        charts_html = ""
        chart_titles = {
            "prediction_distribution": "예측 분포",
            "confidence_distribution": "신뢰도 분포",
            "model_comparison": "모델 비교"
        }
        
        for chart_key, chart_path in charts.items():
            if chart_key != "performance":  # 성능 차트는 이미 위에서 표시
                title = chart_titles.get(chart_key, chart_key)
                rel_path = os.path.relpath(chart_path, self.output_dir)
                charts_html += f"""
                <div class="chart-container">
                    <h3>{title}</h3>
                    <img src="{rel_path}" alt="{title}">
                </div>"""
        
        return f"""
        <div class="section">
            <h2>📈 분석 차트</h2>
            {charts_html}
        </div>"""
    
    def _create_performance_table_html(self, performance_data: Dict) -> str:
        """성능 테이블 HTML 생성"""
        if not performance_data:
            return "<p>성능 데이터가 없습니다.</p>"
        
        table_html = """
        <table>
            <tr>
                <th>모델</th>
                <th>정확도</th>
                <th>정밀도</th>
                <th>재현율</th>
                <th>F1-점수</th>
            </tr>"""
        
        for model, metrics in performance_data.items():
            if isinstance(metrics, dict):
                accuracy = f"{metrics.get('accuracy', 0):.3f}"
                precision = f"{metrics.get('precision', 0):.3f}"
                recall = f"{metrics.get('recall', 0):.3f}"
                f1_score = f"{metrics.get('f1_score', 0):.3f}"
                
                table_html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{accuracy}</td>
                    <td>{precision}</td>
                    <td>{recall}</td>
                    <td>{f1_score}</td>
                </tr>"""
        
        table_html += "</table>"
        return table_html
    
    def _create_predictions_table_html(self, predictions_data: Dict) -> str:
        """예측 테이블 HTML 생성"""
        if not predictions_data:
            return "<p>예측 데이터가 없습니다.</p>"
        
        # 요약 정보 먼저 표시
        summary_html = ""
        if "summary" in predictions_data:
            summary = predictions_data["summary"]
            summary_html = f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total', 0)}</div>
                    <div class="metric-label">총 예측 수</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-safe">{summary.get('safe_count', 0)}</div>
                    <div class="metric-label">안전 예측</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-danger">{summary.get('danger_count', 0)}</div>
                    <div class="metric-label">위험 예측</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value status-unknown">{summary.get('unknown_count', 0)}</div>
                    <div class="metric-label">불명 예측</div>
                </div>
            </div>"""
        
        # 상세 예측 결과 테이블
        table_html = """
        <table>
            <tr>
                <th>이미지</th>
                <th>예측</th>
                <th>신뢰도</th>
                <th>위험도</th>
            </tr>"""
        
        # predictions에서 개별 결과 추출
        if "predictions" in predictions_data:
            for image_path, result in list(predictions_data["predictions"].items())[:20]:  # 최대 20개만 표시
                prediction = result.get("prediction", "unknown")
                confidence = f"{result.get('confidence', 0):.3f}"
                danger_score = f"{result.get('danger_score', 0):.3f}"
                
                status_class = f"status-{prediction}"
                image_name = os.path.basename(image_path)
                
                table_html += f"""
                <tr>
                    <td>{image_name}</td>
                    <td class="{status_class}">{prediction.upper()}</td>
                    <td>{confidence}</td>
                    <td>{danger_score}</td>
                </tr>"""
        
        table_html += "</table>"
        
        if len(predictions_data.get("predictions", {})) > 20:
            table_html += f"<p><em>상위 20개 결과만 표시됨 (전체: {len(predictions_data['predictions'])}개)</em></p>"
        
        return summary_html + table_html
    
    def _extract_summary_stats(self, report_data: Dict) -> List[tuple]:
        """요약 통계 추출"""
        stats = []
        
        # 기본 통계
        if "model_info" in report_data and report_data["model_info"]:
            model_count = len(report_data["model_info"].get("model_types", []))
            stats.append(("사용된 모델 수", str(model_count), "개"))
        
        if "predictions" in report_data and "summary" in report_data["predictions"]:
            total_predictions = report_data["predictions"]["summary"].get("total", 0)
            stats.append(("총 예측 수", str(total_predictions), "개"))
        
        # 성능 통계
        if "performance" in report_data and report_data["performance"]:
            avg_accuracy = np.mean([
                metrics.get("accuracy", 0) 
                for metrics in report_data["performance"].values() 
                if isinstance(metrics, dict)
            ])
            stats.append(("평균 정확도", f"{avg_accuracy:.3f}", ""))
        
        return stats
    
    def _generate_markdown_report(self, report_data: Dict, charts: Dict, output_file: Path):
        """마크다운 보고서 생성"""
        md_content = f"""# {report_data['title']}

**생성 시간:** {report_data['timestamp']}  
**분석 유형:** {report_data['analysis_type']}

---

## 📊 분석 요약

"""
        
        # 요약 통계
        summary_stats = self._extract_summary_stats(report_data)
        for label, value, unit in summary_stats:
            md_content += f"- **{label}:** {value}{unit}\n"
        
        # 모델 정보
        if "model_info" in report_data and report_data["model_info"]:
            md_content += "\n## 🤖 모델 정보\n\n"
            model_info = report_data["model_info"]
            
            if "model_types" in model_info:
                md_content += f"- **사용된 모델:** {', '.join(model_info['model_types'])}\n"
            
            if "is_trained" in model_info:
                status = "✅ 훈련됨" if model_info["is_trained"] else "❌ 미훈련"
                md_content += f"- **훈련 상태:** {status}\n"
        
        # 성능 정보
        if "performance" in report_data and report_data["performance"]:
            md_content += "\n## 🎯 모델 성능\n\n"
            md_content += "| 모델 | 정확도 | 정밀도 | 재현율 | F1-점수 |\n"
            md_content += "|------|--------|--------|--------|----------|\n"
            
            for model, metrics in report_data["performance"].items():
                if isinstance(metrics, dict):
                    accuracy = f"{metrics.get('accuracy', 0):.3f}"
                    precision = f"{metrics.get('precision', 0):.3f}"
                    recall = f"{metrics.get('recall', 0):.3f}"
                    f1_score = f"{metrics.get('f1_score', 0):.3f}"
                    md_content += f"| {model} | {accuracy} | {precision} | {recall} | {f1_score} |\n"
        
        # 예측 결과
        if "predictions" in report_data and report_data["predictions"]:
            md_content += "\n## 🔍 예측 결과\n\n"
            
            if "summary" in report_data["predictions"]:
                summary = report_data["predictions"]["summary"]
                md_content += f"- **총 예측 수:** {summary.get('total', 0)}개\n"
                md_content += f"- **안전 예측:** {summary.get('safe_count', 0)}개\n"
                md_content += f"- **위험 예측:** {summary.get('danger_count', 0)}개\n"
                md_content += f"- **불명 예측:** {summary.get('unknown_count', 0)}개\n"
        
        # 차트 정보
        if charts:
            md_content += "\n## 📈 생성된 차트\n\n"
            for chart_key, chart_path in charts.items():
                rel_path = os.path.relpath(chart_path, self.output_dir)
                chart_name = chart_key.replace('_', ' ').title()
                md_content += f"- **{chart_name}:** `{rel_path}`\n"
        
        md_content += f"\n---\n\n**보고서 생성 시간:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        md_content += "**🔍 SafetyKnob 이미지 안전성 분류 시스템**"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _generate_pdf_report(self, report_data: Dict, charts: Dict, output_file: Path):
        """PDF 보고서 생성"""
        if not WEASYPRINT_AVAILABLE:
            print("PDF 생성을 위해 weasyprint 패키지가 필요합니다: pip install weasyprint")
            return
        
        # HTML 먼저 생성
        html_file = self.output_dir / "temp_report.html"
        self._generate_html_report(report_data, charts, html_file)
        
        # PDF로 변환
        try:
            weasyprint.HTML(filename=str(html_file)).write_pdf(str(output_file))
            os.remove(html_file)  # 임시 HTML 파일 삭제
        except Exception as e:
            print(f"PDF 생성 중 오류: {e}")


def create_analysis_report(analysis_results: Dict, output_dir: str, report_type: str = "all") -> Dict:
    """분석 결과를 보고서로 생성하는 메인 함수"""
    generator = ReportGenerator(output_dir)
    return generator.generate_complete_report(analysis_results, report_type)