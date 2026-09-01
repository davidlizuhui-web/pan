from html.parser import HTMLParser
import unittest
import warnings

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)
from fastapi.testclient import TestClient

from app import app


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])


class FrontendContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_index_exposes_all_interactive_controls(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

        parser = IdCollector()
        parser.feed(response.text)
        required_ids = {
            "parameter-form",
            "x0-input",
            "v0-input",
            "duration-input",
            "calculate-button",
            "play-button",
            "pause-button",
            "reset-button",
            "speed-select",
            "time-slider",
            "time-value",
            "position-value",
            "velocity-value",
            "error-message",
            "simulation-svg",
            "spring-path",
            "mass-block",
            "plot-svg",
            "trajectory-path",
            "time-guide",
            "current-point",
        }
        self.assertEqual(required_ids - parser.ids, set())

    def test_frontend_assets_are_served(self):
        style = self.client.get("/static/style.css")
        script = self.client.get("/static/app.js")

        self.assertEqual(style.status_code, 200)
        self.assertIn("text/css", style.headers["content-type"])
        self.assertEqual(script.status_code, 200)
        self.assertIn("javascript", script.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
