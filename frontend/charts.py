#!/user/bin/env python3

import altair
import pandas as pd

import parse_csv


def make_scatterplot_base(data, color_key):
    altair.renderers.enable("html")

    items = []
    for x in data:
        answer = x["rec"].get(color_key, "Unknown")
        items.extend(
            [
                {
                    "Group": group_value,
                    "full_answer": answer,
                    "x": x["vec"][0],
                    "y": x["vec"][1],
                    "Content": x["sentence"],
                }
                for group_value in parse_csv.split_values(answer)
            ]
        )
    df = pd.DataFrame(items)

    chart = (
        altair.Chart(df, height=400, width=600)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=altair.X("x", axis=None, scale=altair.Scale(zero=False)),
            y=altair.Y("y", axis=None, scale=altair.Scale(zero=False)),
            color="Group",
            tooltip=["Group", "Content"],
        )
        .interactive()
        .configure_view(strokeOpacity=1)
    )
    chart.configure_legend(
        strokeColor="gray",
        fillColor="#EEEEEE",
        padding=10,
        cornerRadius=10,
    )
    return chart
