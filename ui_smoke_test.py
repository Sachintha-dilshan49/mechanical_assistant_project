"""
ui_smoke_test.py
Runs app.py headlessly and fails if any Streamlit exception is raised.

Catches the class of bug a syntax check cannot: a bad st.* call, a missing
session-state key, or a render path that only breaks once there is a message
on screen. Run it after touching the UI.

    py ui_smoke_test.py
"""
import sys
from streamlit.testing.v1 import AppTest


def check(label, at):
    if at.exception:
        print(f"  FAIL  {label}")
        for e in at.exception:
            print(f"        {e.value}")
        return False
    print(f"  ok    {label}")
    return True


at = AppTest.from_file("app.py", default_timeout=120)
at.run()
passed = check("empty state renders", at)

examples = [b for b in at.button if "boat fitting" in b.label]
if not examples:
    print("  FAIL  example prompt buttons missing")
    passed = False
else:
    examples[0].click().run()
    passed &= check("answer renders after clicking a prompt", at)
    msgs = at.session_state["conversations"][at.session_state["current"]]["messages"]
    if len(msgs) < 2:
        print("  FAIL  expected a user message and an answer")
        passed = False
    else:
        print(f"  ok    conversation has {len(msgs)} turns")

print("\nPASS" if passed else "\nFAIL")
sys.exit(0 if passed else 1)
