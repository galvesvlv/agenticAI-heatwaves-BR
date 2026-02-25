# visual_reasoning.py

# imports
from PIL import Image

def visual_reasoning(
                     agent,
                     png_path: str,
                     ) -> str:
    """
    Deterministic visual interpretation of a heatwave prediction map
    using a vision-language agent grounded on authoritative references.

    The agent may call retrieval tools as needed to ground its interpretation
    (e.g., WMO definitions or contextual references).
    """

    prompt = """
Describe the visible spatial patterns in this heatwave prediction map.

Focus on:
- regional hotspots
- spatial gradients
- contrasts between regions

Use cautious language (may, could, potentially).
Do not include numbers.
"""

    img = Image.open(png_path).convert("RGB")
    response = agent.run(
                         prompt,
                         images=[img],
                         )

    return response
