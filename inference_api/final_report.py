# final_report.py

# imports
import json
from inference_api.def_wmo import WMO_DEF

def generate_final_report(
                          model,
                          visual_text: str,
                          stats_path: str,
                          ) -> str:


    with open(stats_path) as f:
        stats = json.load(f)

    prompt = f"""
You are writing a scientific heatwave forecast report.

IMPORTANT:
- Do NOT use print().
- Do NOT write code.
- Write a single, continuous scientific text.
- Use section headers exactly as specified.
- Return only the final report text.
- Do not wrap the output in any function.

WMO definition:
Using the official World Meteorological Organization (WMO) definition
and guidance on heatwaves in:
{WMO_DEF['concept']}

Visual interpretation:
{visual_text}

State-level statistics (JSON):
{json.dumps(stats, indent=2)}

Requirements:
- Reference the WMO definition explicitly
- Cite the 1961–1990 climatological baseline
- State clearly that results are model-based forecasts
- Mention uncertainty implicitly
- Structure the text as:
- Overview
- Spatial Patterns
- State-Level Highlights
- Scientific Context
- Limitations

Return the final report text only.
"""
    response = model.generate(
                              messages=[{"role": "user", "content": prompt}],
                              max_tokens=800,
                              temperature=0.3,
                              )

    return response
