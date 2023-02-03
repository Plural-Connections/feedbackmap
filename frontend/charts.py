#!/user/bin/env python3

import altair
import pandas as pd


def make_scatterplot_base(group_results):
    altair.renderers.enable("html")
    items = []
    for group, results in group_results.items():
        for x in results["matches"]:
            items.append(
                {
                    "Community": group,
                    "x": x["vec"][0],
                    "y": x["vec"][1],
                    "Content": x["sentence"],
                }
            )
    df = pd.DataFrame(items)

    chart = (
        altair.Chart(df, height=400, width=600)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=altair.X("x", axis=None, scale=altair.Scale(zero=False)),
            y=altair.Y("y", axis=None, scale=altair.Scale(zero=False)),
            tooltip=["Community", "Content"],
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
