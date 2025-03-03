# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/game_actions.log"),
    ],
)
logger = logging.getLogger(__name__)
