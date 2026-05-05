import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_ui = False
for i, line in enumerate(lines):
    if line.startswith('# --- UI Components ---'):
        in_ui = True
        new_lines.append(line)
        new_lines.append('st.sidebar.title("App Mode")\n')
        new_lines.append('app_mode = st.sidebar.radio("Select Mode", ["Classic Finder", "Director (v0.2)"])\n\n')
        new_lines.append('def render_classic_mode():\n')
        continue
        
    if in_ui:
        new_lines.append('    ' + line if line.strip() else '\n')
    else:
        new_lines.append(line)

new_lines.append('\n')
new_lines.append('if app_mode == "Classic Finder":\n')
new_lines.append('    render_classic_mode()\n')
new_lines.append('elif app_mode == "Director (v0.2)":\n')
new_lines.append('    from core.director import generate_shot_list\n')
new_lines.append('    from core.director_search import fetch_director_footage\n')
new_lines.append('    from core.output import generate_fcpxml, generate_shot_list_txt\n')
new_lines.append('    st.title("🎬 B-Roll Director (v0.2)")\n')
new_lines.append('    st.write("WIP")\n')

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
