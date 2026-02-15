from emotisense.app import EmotionDetectionApp
from emotisense.config import Config


def run():
    app = EmotionDetectionApp(Config())
    app.run()


if __name__ == "__main__":
    run()
