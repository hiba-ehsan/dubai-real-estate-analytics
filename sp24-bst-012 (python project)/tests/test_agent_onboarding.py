import io
import os


def test_agent_instructions_exists():
    path = os.path.join(os.path.dirname(__file__), os.pardir, 'AGENT_INSTRUCTIONS.md')
    path = os.path.abspath(path)
    assert os.path.exists(path), "AGENT_INSTRUCTIONS.md should exist"
    content = open(path, 'r', encoding='utf-8').read()
    assert 'Agent Onboarding' in content or 'Purpose' in content, "AGENT_INSTRUCTIONS.md should include onboarding purpose or headings"


def test_readme_mentions_child_friendly():
    readme_path = os.path.join(os.path.dirname(__file__), os.pardir, 'README_AIRBNB.md')
    readme_path = os.path.abspath(readme_path)
    content = open(readme_path, 'r', encoding='utf-8').read()
    assert "Explain like I'm 10" in content or "child-friendly" in content, "README should mention the child-friendly explanation"


def test_price_prediction_contains_child_expander():
    page_path = os.path.join(os.path.dirname(__file__), os.pardir, 'pages', 'Price Prediction.py')
    page_path = os.path.abspath(page_path)
    content = open(page_path, 'r', encoding='utf-8').read()
    assert "Explain like I'm 10" in content, "Price Prediction page should include the child-friendly expander title"
