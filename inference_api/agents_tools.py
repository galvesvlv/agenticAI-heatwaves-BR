# agents_tools.py

# imports
from smolagents import tool, WebSearchTool

@tool
def retrieve_WMO_definition() -> dict:
    """
    Retrieve the World Meteorological Organization (WMO) definition
    and guidelines for heatwaves.

    This tool provides an authoritative textual reference used as
    grounding knowledge for interpretation of heatwave predictions.
    The content is static, curated, and suitable for retrieval-augmented
    generation (RAG) workflows.

    Returns
    -------
    dict
        Dictionary containing the official WMO heatwave definition,
        methodological notes, and reference source.
    """

    definition = {
                  "organization": "World Meteorological Organization (WMO)",
                  "concept": (
                              "Days with heatwave conditions correspond to days "
                              "on which the daily maximum temperature exceeds "
                              "the mean monthly maximum temperature by at least 5 °C."
                             ),
                  "criteria": {
                  "temperature_reference": (
                                            "Heatwaves are defined relative to local climatology, "
                                            "typically using percentile-based or anomalies thresholds related to "
                                            "heat."
                                            ),
                               },
                  "regional_context": (
                                       "However, The WMO emphasizes that heatwave definitions must be region-specific, "
                                       "as impacts and thermal comfort depend on local climate, population "
                                       "adaptation, and environmental conditions."
                                       ),
                  "notes": (
                            "However, the WMO itself emphasizes the lack of a "
                             "single universal definition and notes that, more broadly, "
                             "heatwaves are associated with a period of excessive heat "
                             "accumulated over days and nights that are warmer than normal."
                            ),
                  "reference": (
                                "Instituto Nacional de Meteorologia. (n.d.). Glossário: Onda de calor. "
                                "From https://portal.inmet.gov.br/glossario/glossario#O."
                                "World Meteorological Organization. 2023. "
                                "Guidelines on the Definition and Characterization of Extreme Weather and Climate Events. "
                                "WMO-No. 1310. Geneva: World Meteorological Organization. "
                                "https://library.wmo.int/viewer/58396/download?file=1310_Guidelines_on_DEWCE_en.pdf&navigator=1&type=pdf"
                                )
                }

    return definition

@tool
def retrieve_contextual_references(query: str) -> dict:
    """
    Retrieve contextual scientific or institutional references from the web.

    This tool is intended to support retrieval-augmented generation (RAG)
    workflows by providing external contextual information when needed.

    Args:
        query (str): Search query describing the scientific or institutional
            context to retrieve (e.g., heatwaves, climate extremes,
            WMO guidelines).

    Returns:
        dict: Search results returned by the web search tool, which may include
            institutional reports, official guidelines, or peer-reviewed
            scientific references.
    """
    internet_search = WebSearchTool()
    
    return internet_search(query)