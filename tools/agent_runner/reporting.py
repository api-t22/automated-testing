from typing import List, Dict, Any
from datetime import datetime
from xlsxwriter.utility import xl_range

class ReportGenerator:
    """
    Generates structured QA reports with metrics and detailed issue tables.
    """
    
    SEVERITY_MAP = {
        "P0": "Blocking Issue/Crash",
        "P1": "Major Impact on Functionality",
        "P2": "Minor Impact on Functionality",
        "P3": "Cosmetic Issue/Typo"
    }

    def __init__(self, results: List[Dict[str, Any]], plan_data: Dict[str, Any] = None):
        self.results = results
        self.plan_data = plan_data or {}
        self.failed_tests = [r for r in results if r["status"] == "FAIL"]
        self.passed_tests = [r for r in results if r["status"] == "PASS"]

    def generate_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary counts for each issue severity category.
        """
        metrics = {
            "Blocking Issue/Crash": 0,
            "Major Impact on Functionality": 0,
            "Minor Impact on Functionality": 0,
            "Cosmetic Issue/Typo": 0,
            "Total Failed": len(self.failed_tests),
            "Total Passed": len(self.passed_tests),
            "Total Tests": len(self.results)
        }

        # Map failures to severity
        for res in self.failed_tests:
            # Look up priority from plan data if available, or default
            prio = res.get("priority", "P1") 
            severity = self.SEVERITY_MAP.get(prio, "Major Impact on Functionality")
            metrics[severity] = metrics.get(severity, 0) + 1
            
        return metrics

    def generate_markdown(self) -> str:
        """
        Generate the full markdown report string.
        """
        metrics = self.generate_metrics()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = []
        md.append(f"# QA Test Report")
        md.append(f"**Date:** {now}\n")

        # 1. Metrics Section
        md.append("## Executive Summary")
        md.append("| Metric | Count |")
        md.append("| :--- | :---: |")
        md.append(f"| **Blocking Issue/Crash** | {metrics['Blocking Issue/Crash']} |")
        md.append(f"| **Major Impact** | {metrics['Major Impact on Functionality']} |")
        md.append(f"| **Minor Impact** | {metrics['Minor Impact on Functionality']} |")
        md.append(f"| **Cosmetic Issue** | {metrics['Cosmetic Issue/Typo']} |")
        md.append(f"| | |")
        md.append(f"| Total Passed | {metrics['Total Passed']} |")
        md.append(f"| Total Failed | {metrics['Total Failed']} |")
        md.append("\n")

        # 2. Detailed Issues Table
        md.append("## Detailed Issues Table")
        if not self.failed_tests:
            md.append("\n*No issues found. All tests passed.* 🎉\n")
        else:
            headers = ["Issue ID", "Severity", "Description", "Steps", "Expected", "Actual", "Status"]
            md.append("| " + " | ".join(headers) + " |")
            md.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for res in self.failed_tests:
                issue_id = res.get("id", "UNK")
                prio = res.get("priority", "P1")
                severity = self.SEVERITY_MAP.get(prio, "Major")
                # Description comes from Title
                desc = res.get("title", "")
                
                # Steps (Formatted as list)
                steps_list = res.get("steps", [])
                if isinstance(steps_list, list):
                    steps_md = "<br>".join([f"{i+1}. {s}" for i, s in enumerate(steps_list)])
                else:
                    steps_md = str(steps_list)

                expected = res.get("expected_result", "")
                # Actual comes from error message usually
                actual = res.get("error", "Unknown Error")
                status = "Open"
                
                row = [
                    issue_id,
                    severity,
                    desc,
                    steps_md,
                    expected,
                    actual,
                    status
                ]
                # Escape pipes
                row = [str(c).replace("|", "\\|") for c in row]
                md.append("| " + " | ".join(row) + " |")
        
        return "\n".join(md)

    def generate_xlsx(self, filename: str):
        """
        Generate a multi-tab Excel report with charts.
        """
        import pandas as pd
        
        # 1. Prepare DataFrames
        metrics = self.generate_metrics()
        summary_rows = [
            {"Category": "Blocking Issue/Crash", "Count": metrics["Blocking Issue/Crash"]},
            {"Category": "Major Impact", "Count": metrics["Major Impact on Functionality"]},
            {"Category": "Minor Impact", "Count": metrics["Minor Impact on Functionality"]},
            {"Category": "Cosmetic Issue", "Count": metrics["Cosmetic Issue/Typo"]},
            {"Category": "Total Passed", "Count": metrics["Total Passed"]},
            {"Category": "Total Failed", "Count": metrics["Total Failed"]},
        ]
        summary_df = pd.DataFrame(summary_rows)

        issues_data = []
        for res in self.failed_tests:
            prio = res.get("priority", "P1")
            severity = self.SEVERITY_MAP.get(prio, "Major Impact")
            steps = res.get("steps", [])
            steps_str = "\\n".join(steps) if isinstance(steps, list) else str(steps)
            
            issues_data.append({
                "Issue ID": res.get("id"),
                "Severity": severity,
                "Description": res.get("title"),
                "Steps": steps_str,
                "Expected": res.get("expected_result"),
                "Actual Error": res.get("error"),
                "Status": "Open"
            })
        issues_df = pd.DataFrame(issues_data)
        
        results_df = pd.DataFrame(self.results)
        
        stories_data = []
        stories = self.plan_data.get("user_stories", [])
        for s in stories:
            stories_data.append({
                "Story ID": s.get("id"),
                "Title": s.get("title"),
                "Goal": s.get("goal"),
                "Covered By": ", ".join(s.get("related_test_ids", []))
            })
        stories_df = pd.DataFrame(stories_data)

        # 2. Write to Excel
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Dashboard', index=False)
            if not issues_df.empty:
                issues_df.to_excel(writer, sheet_name='Issues List', index=False)
            results_df.to_excel(writer, sheet_name='All Test Results', index=False)
            if not stories_df.empty:
                stories_df.to_excel(writer, sheet_name='User Stories', index=False)
                
            # 3. Add Charts
            # 3. Define Formats
            workbook = writer.book
            header_fmt = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#2563EB',
                'font_color': '#FFFFFF',
                'border': 1
            })
            cell_fmt = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })
            status_pass_fmt = workbook.add_format({
                'bg_color': '#93C47D',
                'font_color': '#000000',
                'border': 1
            })
            status_fail_fmt = workbook.add_format({
                'bg_color': '#E06666',
                'font_color': '#000000',
                'border': 1
            })
            status_warn_fmt = workbook.add_format({
                'bg_color': '#FFD966',
                'font_color': '#000000',
                'border': 1
            })
            prio_formats = {
                "P0": workbook.add_format({'bg_color': '#E06666', 'font_color': '#000000', 'border': 1}),
                "P1": workbook.add_format({'bg_color': '#F6B26B', 'font_color': '#000000', 'border': 1}),
                "P2": workbook.add_format({'bg_color': '#FFD966', 'font_color': '#000000', 'border': 1}),
                "P3": workbook.add_format({'bg_color': '#93C47D', 'font_color': '#000000', 'border': 1}),
            }
            
            # Helper to apply format to sheet
            def format_sheet(sheet_name, df):
                if sheet_name not in writer.sheets: return
                ws = writer.sheets[sheet_name]
                # Apply header format
                for col_num, value in enumerate(df.columns.values):
                    ws.write(0, col_num, value, header_fmt)
                
                # Apply Column Widths (Approx)
                ws.set_column(0, len(df.columns) - 1, 20, cell_fmt) # Default width 20 + wrap
                
                # Specific fine tuning
                if sheet_name == 'Dashboard':
                    ws.set_column('A:A', 30, cell_fmt)
                    ws.set_column('B:B', 15, cell_fmt)
                elif sheet_name == 'Issues List':
                    ws.set_column('A:B', 15, cell_fmt) # ID, Severity
                    ws.set_column('C:C', 40, cell_fmt) # Description
                    ws.set_column('D:F', 30, cell_fmt) # Steps, Exp, Act
                elif sheet_name == 'User Stories':
                    ws.set_column('A:A', 15, cell_fmt)
                    ws.set_column('B:C', 40, cell_fmt) # Title, Goal
                    ws.set_column('D:D', 30, cell_fmt) # Covered By

            # 4. Apply Formatting
            format_sheet('Dashboard', summary_df)
            if not issues_df.empty: format_sheet('Issues List', issues_df)
            format_sheet('All Test Results', results_df)
            if not stories_df.empty: format_sheet('User Stories', stories_df)

            def apply_conditional_formatting(sheet_name: str, df):
                if sheet_name not in writer.sheets:
                    return
                ws = writer.sheets[sheet_name]
                if df.empty:
                    return
                columns = list(df.columns)
                row_count = len(df)

                # Status coloring
                status_idx = next((i for i, c in enumerate(columns) if c.lower() == "status"), None)
                if status_idx is not None:
                    status_range = xl_range(1, status_idx, row_count, status_idx)
                    ws.conditional_format(status_range, {
                        'type': 'cell',
                        'criteria': '==',
                        'value': '"PASS"',
                        'format': status_pass_fmt
                    })
                    ws.conditional_format(status_range, {
                        'type': 'cell',
                        'criteria': '==',
                        'value': '"FAIL"',
                        'format': status_fail_fmt
                    })
                    for pending_val in ("Open", "OPEN"):
                        ws.conditional_format(status_range, {
                            'type': 'cell',
                            'criteria': '==',
                            'value': f'"{pending_val}"',
                            'format': status_warn_fmt
                        })

                # Priority coloring
                prio_idx = next((i for i, c in enumerate(columns) if c.lower() == "priority"), None)
                if prio_idx is not None:
                    prio_range = xl_range(1, prio_idx, row_count, prio_idx)
                    for key, fmt in prio_formats.items():
                        ws.conditional_format(prio_range, {
                            'type': 'cell',
                            'criteria': '==',
                            'value': f'"{key}"',
                            'format': fmt
                        })

            apply_conditional_formatting('All Test Results', results_df)
            if not issues_df.empty:
                apply_conditional_formatting('Issues List', issues_df)

            # 5. Add Charts (Dashboard)
            worksheet = writer.sheets['Dashboard']
            
            # Create Pie Chart for Pass/Fail
            chart = workbook.add_chart({'type': 'pie'})
            chart.add_series({
                'name': 'Pass/Fail Ratio',
                'categories': '=Dashboard!$A$6:$A$7',
                'values':     '=Dashboard!$B$6:$B$7',
                'points': [
                    {'fill': {'color': '#10B981'}}, # Green for Pass
                    {'fill': {'color': '#EF4444'}}, # Red for Fail
                ],
            })
            chart.set_title({'name': 'Test Execution Status'})
            worksheet.insert_chart('D2', chart)

            # Create Bar Chart for Severity
            severity_chart = workbook.add_chart({'type': 'column'})
            severity_chart.add_series({
                'name': 'Severity',
                'categories': '=Dashboard!$A$2:$A$5',
                'values':     '=Dashboard!$B$2:$B$5',
                'fill':       {'color': '#F59E0B'} 
            })
            severity_chart.set_title({'name': 'Defect Severity Distribution'})
            worksheet.insert_chart('D18', severity_chart)
