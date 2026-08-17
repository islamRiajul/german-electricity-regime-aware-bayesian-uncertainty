import os

figures_dir = "figures"
html_files = []

if os.path.exists(figures_dir):
    for f in sorted(os.listdir(figures_dir)):
        if f.endswith(".html"):
            html_files.append(f)

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>German Electricity Research - Interactive Plots</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f4f6f9; color: #333; }
        h1 { color: #2c3e50; }
        p { color: #555; }
        ul { list-style-type: none; padding: 0; }
        li { margin: 10px 0; background: #fff; padding: 12px 18px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 650px; }
        a { text-decoration: none; color: #2980b9; font-size: 16px; font-weight: bold; display: block; }
        a:hover { color: #1abc9c; }
    </style>
</head>
<body>
    <h1>German Electricity Research - Interactive Plots</h1>
    <p>Explore all interactive model outputs and forecast dashboards below:</p>
    <ul>
"""

for filename in html_files:
    title = filename.replace(".html", "").replace("_", " ").title()
    file_path = f"figures/{filename}"
    html_content += f'        <li><a href="{file_path}" target="_blank">📊 {title}</a></li>\n'

html_content += """    </ul>
</body>
</html>
"""

with open("index.html", "w") as f:
    f.write(html_content)

print(f"Successfully found {len(html_files)} HTML files in 'figures/' and updated index.html!")
