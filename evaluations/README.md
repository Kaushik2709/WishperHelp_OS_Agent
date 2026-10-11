# Safe computer-use task evaluations

Run from the repository root. These evaluations use no external APIs, microphone,
screenshots, personal documents, payments, or destructive operations.

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/ -v
python -m evaluations.run --mode mocked --output evaluation-results/mocked.json
python -m evaluations.run --mode filesystem --output evaluation-results/filesystem.json
```

The mocked demo launches an in-memory application, enters Hindi/emoji text, clicks
a known control, handles a missing element, uses one explicitly approved alternate
control, and cancels between steps. It exercises the real action executor against
a simulated HAL. It does not test the microphone, a model, or a desktop.

Filesystem mode creates, appends, and reads a disposable Unicode document using
the real filesystem. It also attempts an overwrite and verifies that confirmation
holds the action while preserving the original content. The desktop HAL is simulated.

For a genuine Windows desktop attempt, use a dedicated interactive test session:

```powershell
python -m pip install -r requirements.txt
python -m evaluations.run --mode desktop --output evaluation-results/desktop.json
```

The harness starts its own PySide6 fixture, makes at most one attempt to focus its
exact title, and checks the fixture PID/title immediately before every input call.
Windows UIA, pywin32, a usable interactive desktop, and equivalent integrity levels
for the runner and fixture are required. The test oracle is the fixture's JSON state,
not a model's claim or screen difference. Typing is only into the isolated editor;
navigation only changes a label in that window. A failure to verify foreground
stops input. The fixture exits via a local stop file, with a 120-second lifetime
bound; only its own process can be terminated during cleanup.

macOS/Linux desktop mode records `not_run`; their native GUI harnesses remain to
be implemented. Headless CI is not desktop validation, even on an OS-specific runner.

Each output records OS/Python, timestamps, completion (`successful`, `partial`,
`failed`, `not_run`), independently checked postconditions, action calls, native
input calls, file mutations attempted, latency, recovery attempts, blocked risky
attempts, and failure codes. `expected_behavior_met` also counts correctly stopped
negative scenarios; it must not be presented as the GUI task completion rate.
Startup is separately labeled as harness setup, not production `launch_app` coverage.

Artifacts stay inside the repository, temporary fixtures use unique subdirectories,
and cleanup only removes those directories. Use `evaluations/results/` for retained
review evidence; `evaluation-results/` is ignored scratch output.
