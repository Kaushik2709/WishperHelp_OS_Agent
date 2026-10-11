import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))
from wisprhelp.models.runtime import local_runtime

print("Ensuring local in-RAM llama-server is running...")
if not local_runtime.is_server_reachable():
    local_runtime.prewarm_reasoning_model()

system_prompt = '''You are WisprHelp, a fast local computer-use planner running on the user's windows workstation.
Decompose user instructions into ordered atomic actions.

ACTIONS:
- launch_app: open application (target = app name e.g. "Notepad", "Chrome", "WhatsApp")
- open_folder: open folder in file explorer (target = folder path e.g. "Documents", "Downloads")
- open_file: open file in desktop app (target = file path or "ordinal:0:file", params = {"directory": "Documents"})
- click: click UI element (target = element name or "ordinal:0:chat")
- double_click: double click element (target = element name or "ordinal:0:file")
- create_file: create a new file (target = file path, payload = file content)
- type_text: type text into focused field (payload = exact text)
- respond: chat answer without OS action (payload = response)

Output STRICTLY valid JSON with structure:
{
  "primary_goal": "<summary>",
  "is_conversational": false,
  "steps": [
    {"action": "<action>", "target": "<target>", "payload": "<payload>", "params": {}, "condition": null, "is_destructive": false, "confidence": 0.95}
  ]
}'''

queries = [
    'click on the first chat on WhatsApp',
    'click on the first set on WhatsApp',
    'Open Documents and open the first file',
    'Open Documents and open the second file'
]

for q in queries:
    resp = requests.post(
        'http://127.0.0.1:8088/v1/chat/completions',
        json={
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'User Instruction: "{q}"\nDecompose into JSON steps now:'}
            ],
            'temperature': 0.1,
            'max_tokens': 300,
            'response_format': {'type': 'json_object'}
        },
        timeout=30
    )
    print('QUERY:', q)
    print(resp.json()['choices'][0]['message']['content'])
    print('-'*50)
