import pandas as pd
import streamlit as st
import zipfile
from xml.etree import ElementTree as ET


@st.cache_data
def load_edges(path="UserEdges.xlsx"):
    try:
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows_data = []

        with zipfile.ZipFile(path, "r") as z:
            with z.open("xl/worksheets/sheet1.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()
                for row in root.findall(".//x:row", ns):
                    cells = row.findall("x:c", ns)
                    vals = []
                    for c in cells:
                        # inlineStr
                        t = c.find(".//x:is/x:t", ns)
                        if t is not None:
                            vals.append(t.text)
                        else:
                            v = c.find("x:v", ns)
                            vals.append(v.text if v is not None else None)
                    rows_data.append(vals)

        df = pd.DataFrame(rows_data[1:], columns=rows_data[0])
        return df

    except Exception as e:
        st.warning(f"UserEdges tidak bisa dibaca: {e}")
        return pd.DataFrame(columns=["source", "target"])