#!/user/bin/env python3

from collections import defaultdict

import altair
import pandas as pd

import parse_csv


def make_scatterplot_base(data, color_key):
    altair.renderers.enable("html")
    value_counts = defaultdict(lambda: 0)
    items = []
    for x in data:
        answer = x["rec"].get(color_key, "Unknown")
        answer_values = parse_csv.split_values(answer)
        for v in answer_values:
            value_counts[v] += 1
        items.extend(
            [
                {
                    "ColorGroup": group_value,
                    "Groups": answer,
                    "x": x["vec"][0],
                    "y": x["vec"][1],
                    "Response": x["sentence"],
                }
                for group_value in answer_values
            ]
        )

    # Display the value count in the legend
    legend_values = {}
    for item in items:
        legend_value = "%s [%0.2f%%]" % (
            item["ColorGroup"],
            100.0 * value_counts[item["ColorGroup"]] / len(data),
        )
        legend_values[legend_value] = value_counts[item["ColorGroup"]]
        item["GroupForLegend"] = legend_value
    df = pd.DataFrame(items)

    # Sort legend values by frequency (descending) so that most frequent are
    # on the top of the legend.
    sorted_values = list(legend_values.keys())
    sorted_values.sort(key=lambda x: legend_values[x], reverse=True)

    chart = (
        altair.Chart(df, height=400, width=800)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=altair.X("x", axis=None, scale=altair.Scale(zero=False)),
            y=altair.Y("y", axis=None, scale=altair.Scale(zero=False)),
            color=altair.Color("GroupForLegend", sort=sorted_values),
            tooltip=["Response", "Groups"],
        )
        .interactive()
        .configure_view(strokeOpacity=1)
    ).configure_legend(
        strokeColor="gray",
        fillColor="#EEEEEE",
        orient="right",
        title=None,
        labelLimit=300,
        cornerRadius=10,
    )
    return chart
